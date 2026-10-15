"""Deliver reviewed WATCH cases without resending old failed imports."""
import asyncio
from pathlib import Path
import secrets
import sqlite3
import time
import uuid

from core import read_source, StopBridge
from telegram_client import Flood


def source(c):
    db = sqlite3.connect(Path(c.config['source_db']).resolve(strict=True).as_uri()+'?mode=ro', uri=True, timeout=3)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    db.execute('PRAGMA trusted_schema=OFF')
    return db


def candidates(c, *, include_pending=False, notify=True):
    snapshot = read_source(c.config['source_db'], c.users() | {c.config['account_id'], c.config['hex_bot_id'], c.config['spb_bot_id']})
    db = source(c)
    try:
        approved = db.execute("""SELECT c.id,c.target_user_id FROM cases c JOIN review_updates r ON r.case_id=c.id
            WHERE c.matched_rule_pattern='WATCH_SPY' AND c.netban_eligible=1
            AND c.action='report_approved' AND c.status='approved_and_banned'
            AND r.kind='report' AND r.decision='approve' AND r.actor_id=c.actor_user_id
            AND r.chat_id=? ORDER BY c.created_at,c.id""", (c.config['destination_chat_id'],)).fetchall()
        for row in db.execute("""SELECT d.case_id FROM report_deliveries d JOIN cases c ON c.id=d.case_id
            WHERE c.matched_rule_pattern='WATCH_SPY' AND d.state='pending'
            AND (d.last_error IS NOT NULL OR (d.attempts>1 AND d.review_message_id IS NULL))"""):
            if not notify:
                continue
            with c.db:
                for chat in (c.owner,c.config['destination_chat_id']):
                    c.enqueue(f"spy-report-error:{row[0]}:{chat}",chat,f"⚠️ WATCH 案件 #{row[0].replace('-','')[:8]} 未能送出審核卡，尚未加入黑名單。")
    finally:
        db.close()
    active_users = {v['user_id'] for v in snapshot['cases'].values()}
    # Remove only entries whose successful WATCH delivery is in our own ledger.
    owned = c.db.execute("""SELECT DISTINCT o.user_id,s.case_id FROM owned o JOIN operations p ON p.user_id=o.user_id
        JOIN spy_operations s ON s.operation_id=p.id WHERE o.active=1 AND p.action='ban' AND p.state='acked'""").fetchall()
    changes = [('unban', r['user_id'], r['case_id']) for r in owned if r['user_id'] not in active_users]
    for row in approved:
        if row['id'] not in snapshot['cases'] or not c.db.execute('SELECT 1 FROM watch_spy WHERE id=?',(row['id'],)).fetchone():
            continue
        if c.db.execute('SELECT 1 FROM owned WHERE user_id=? AND active=1',(row['target_user_id'],)).fetchone():
            continue
        if not include_pending and c.db.execute("SELECT 1 FROM spy_operations s JOIN operations o ON o.id=s.operation_id WHERE s.case_id=? AND o.action='ban' AND o.state!='cancelled'",(row['id'],)).fetchone():
            continue
        changes.append(('ban',row['target_user_id'],row['id']))
    return changes


def claim(c, change, now=None):
    now = int(time.time()) if now is None else now
    action,user,case = change
    # Only a recorded, final receiver rejection from an old import can be left
    # aside. An uncertain send is never bypassed, even in the new queue.
    blocked = c.db.execute("""SELECT 1 FROM operations o WHERE o.state NOT IN ('acked','cancelled','rejected')
        AND (o.user_id=? OR COALESCE(o.error,'')!='receiver_failed' OR o.reply_id IS NULL
             OR EXISTS(SELECT 1 FROM spy_operations s WHERE s.operation_id=o.id)) LIMIT 1""",(user,)).fetchone()
    if blocked:
        raise StopBridge('An earlier operation needs reconciliation')
    recent = [r[0] for r in c.db.execute('SELECT created_at FROM operations WHERE created_at>? ORDER BY created_at DESC',(now-86400,))]
    if (recent and now-recent[0]<20) or sum(t>now-3600 for t in recent)>=30 or len(recent)>=150 or now<c.ledger.get('not_before',0):
        return None
    if action not in ('ban','unban') or not 0<user<2**52:
        raise StopBridge('Invalid WATCH operation')
    reference = str(uuid.uuid4())
    command = f"/{action} {user} +spb SPB #{uuid.UUID(case).hex[:8]} sync:{uuid.UUID(reference).hex[:12]}"
    with c.db:
        c.db.execute("INSERT INTO operations(id,user_id,action,case_id,command,random_id,state,created_at) VALUES (?,?,?,?,?,?,'sending',?)",
                     (reference,user,action,case,command,secrets.randbits(63) or 1,now))
        c.db.execute('INSERT INTO spy_operations VALUES (?,?)',(reference,case))
    return dict(c.db.execute('SELECT * FROM operations WHERE id=?',(reference,)).fetchone())


def notices(c):
    rows = c.db.execute("SELECT o.* FROM operations o JOIN spy_operations s ON s.operation_id=o.id WHERE o.state IN ('acked','uncertain')").fetchall()
    with c.db:
        for op in rows:
            ok = op['state']=='acked'
            if ok:
                c.db.execute("UPDATE watch_outbox SET state='cancelled' WHERE key LIKE ? AND state='queued'",(f"spy-result:{op['id']}:uncertain:%",))
            body = (('✅ hexDrill 已確認' + ('加入' if op['action']=='ban' else '移除') + ' +spb。') if ok
                    else '⚠️ hexDrill 未確認成功，WATCH 後續同步已停止。請核對回覆，勿重複送出。')
            body += f"\n使用者：{op['user_id']}\n案件：#{uuid.UUID(op['case_id']).hex[:8]}\nsync:{uuid.UUID(op['id']).hex[:12]}"
            for chat in ((c.config['destination_chat_id'],) if ok else (c.owner,c.config['destination_chat_id'])):
                c.enqueue(f"spy-result:{op['id']}:{op['state']}:{chat}",chat,body)


async def tick(c, t):
    if not c.config.get('receiver_scope_confirmed'):
        raise StopBridge('Receiver permissions have not been confirmed')
    pending = c.db.execute("SELECT o.* FROM operations o JOIN spy_operations s ON s.operation_id=o.id WHERE o.state NOT IN ('acked','cancelled','rejected') ORDER BY o.created_at,o.id LIMIT 1").fetchone()
    if pending:
        await t.reconcile_operation(c.ledger,dict(pending))
        notices(c)
        return
    notices(c)
    changes = candidates(c)
    if not changes:
        return
    await t.check_destination()
    # Check revocation/whitelisting again immediately before a send.
    if changes[0] not in candidates(c):
        return
    try:
        op = claim(c,changes[0])
    except StopBridge:
        with c.db:
            for chat in (c.owner,c.config['destination_chat_id']):
                c.enqueue('spy-blocked:'+str(chat),chat,'⚠️ WATCH 有未確認的舊操作，新核准案件已保留，hexDrill 同步暫停。')
        return
    if op is None:
        return
    try:
        message = await asyncio.wait_for(t.send(op),timeout=25)
        c.ledger.sent(op['id'],message)
    except Flood as error:
        c.ledger.flood(op['id'],error.seconds)
        raise
    except BaseException:
        c.ledger.uncertain(op['id'])
        notices(c)
        raise
