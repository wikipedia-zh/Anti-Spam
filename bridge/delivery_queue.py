"""Explicitly resume approved deliveries, probing the last rejection first."""
import asyncio
import secrets
import time
import uuid

from core import read_source, StopBridge
from telegram_client import Flood


KEY = 'delivery_queue'
ACTIVE = ('requested', 'probe', 'running')
UNFINISHED = "state NOT IN ('acked','cancelled','rejected')"


def state(c):
    return c.ledger.get(KEY, {})


def managed(c):
    return state(c).get('status') in (*ACTIVE, 'paused')


def pending(c):
    return [dict(r) for r in c.db.execute(
        f'SELECT * FROM operations WHERE {UNFINISHED} ORDER BY created_at DESC,rowid DESC')]


def rejected(op):
    return op['error'] == 'receiver_failed' and bool(op['reply_id']) and bool(op['message_id'])


def changes(c):
    from bridge import binding
    snapshot = read_source(c.config['source_db'], c.users() | {
        c.config['account_id'], c.config['hex_bot_id'], c.config.get('spb_bot_id')})
    c.ledger.check(snapshot, binding(c.config))
    approved = {r[0] for r in c.db.execute('SELECT case_id FROM seen WHERE approved=1')}
    owned = {r[0] for r in c.db.execute('SELECT user_id FROM owned WHERE active=1')}
    active = {v['user_id'] for v in snapshot['cases'].values()}
    targets = {}
    for case, item in snapshot['cases'].items():
        if case in approved and item['user_id'] not in owned:
            targets.setdefault(('ban', item['user_id']), ('ban', item['user_id'], case))
    for user in sorted(owned - active):
        row = c.db.execute("SELECT case_id FROM operations WHERE user_id=? AND action='ban' AND state='acked' ORDER BY created_at DESC,rowid DESC LIMIT 1", (user,)).fetchone()
        if not row:
            raise StopBridge('Confirmed ownership has no receipt')
        targets[('unban', user)] = ('unban', user, row[0])
    if len(owned - active) > 20:
        raise StopBridge('More than 20 removals need inspection')
    if c.config.get('spy_enabled'):
        from spy_delivery import candidates
        for change in candidates(c, include_pending=True, notify=False):
            targets[(change[0], change[1])] = change
    return sorted(targets.values(), key=lambda x: (x[0] != 'unban', x[1]))


def status(c):
    todo, waiting = changes(c), pending(c)
    # An in-flight ban may have been reversed at the source; it still needs a receipt.
    total = len({(x[0], x[1]) for x in todo} | {(x['action'], x['user_id']) for x in waiting})
    failed = sum(rejected(x) for x in waiting)
    unknown = len(waiting) - failed
    labels = {'requested': '準備重試', 'probe': '先試一筆', 'running': '寫入中',
              'paused': '已暫停', 'done': '本輪已完成'}
    label = labels.get(state(c).get('status'), '未啟動')
    unsent = len({(x[0], x[1]) for x in todo} - {(x['action'], x['user_id']) for x in waiting})
    text = f'hexDrill +spb 佇列：{label}\n剩餘：{total} 筆\n尚未發送：{unsent} 筆\n已回覆失敗：{failed} 筆\n等候回覆／結果不明：{unknown} 筆'
    if total:
        text += '\n/queue resume：先重試最近失敗的一筆，成功後繼續。'
    cooldown = max(c.ledger.get('not_before', 0), c.ledger.get('telegram_wait_until', 0))
    recent = [r[0] for r in c.db.execute('SELECT created_at FROM operations WHERE created_at>? ORDER BY created_at DESC', (int(time.time()) - 86400,))]
    if len(recent) >= 30:
        cooldown = max(cooldown, recent[29] + 3600)
    if len(recent) >= 150:
        cooldown = max(cooldown, recent[149] + 86400)
    if cooldown > int(time.time()):
        text += f'\n發送冷卻剩餘：{cooldown - int(time.time())} 秒'
    return text


def command(c, chat, message, actor, parts, date):
    if parts == ['/queue']:
        return status(c)
    if parts != ['/queue', 'resume']:
        return '用法：/queue 或 /queue resume'
    if date is None or not 0 <= int(time.time()) - date <= 120:
        return '指令已過期，請重新輸入 /queue resume。'
    if not c.config.get('receiver_scope_confirmed'):
        return '接收端權限尚未確認，未啟動寫入。'
    if state(c).get('status') in ACTIVE:
        return '佇列已啟動；用 /queue 查看剩餘數量。'
    waiting = pending(c)
    if any(not rejected(op) for op in waiting):
        return '有操作尚未確認結果，不能重送。請先用 /sync 核對。'
    todo = changes(c)
    if not waiting and not todo:
        return '沒有待寫入項目。'
    latest = waiting[0] if waiting else None
    probe = (latest['action'], latest['user_id'], latest['case_id']) if latest else todo[0]
    if probe not in todo:
        return '最近失敗案件已撤銷或不再符合收錄條件，未重送；請先核對案件。'
    c.ledger.put(KEY, {'status': 'requested', 'id': f'{chat}:{message}', 'actor': actor,
                       'probe': list(probe), 'operation': None})
    return '先重試一筆，收到 hexDrill 成功回覆後繼續剩餘寫入；再次失敗會暫停並通知。'


def notice(c, run, suffix, body):
    for chat in (c.owner, c.config['destination_chat_id']):
        c.enqueue(f"queue:{run['id']}:{suffix}:{chat}", chat, body)


def pause(c, reason):
    run = state(c)
    if run.get('status') not in ACTIVE:
        return
    run['status'] = 'paused'
    with c.db:
        c.ledger.put(KEY, run)
        c.ledger.put('delivery_paused', True)
        notice(c, run, 'paused', '⚠️ +spb 佇列已暫停。' + reason)


def claim(c, run, change, previous):
    now = int(time.time())
    recent = [r[0] for r in c.db.execute('SELECT created_at FROM operations WHERE created_at>? ORDER BY created_at DESC', (now - 86400,))]
    if ((recent and now - recent[0] < 20) or sum(t > now - 3600 for t in recent) >= 30
            or len(recent) >= 150 or now < c.ledger.get('not_before', 0)):
        return None
    action, user, case = change
    op_id = str(uuid.uuid4())
    text = f'/{action} {user} +spb SPB #{uuid.UUID(case).hex[:8]} sync:{uuid.UUID(op_id).hex[:12]}'
    # Cancel only proven failures, in the same transaction as their replacement.
    with c.db:
        for old in previous:
            updated = c.db.execute("UPDATE operations SET state='cancelled' WHERE id=? AND state='uncertain' AND error='receiver_failed' AND reply_id=?",
                                   (old['id'], old['reply_id'])).rowcount
            if updated != 1:
                raise StopBridge('Retry state changed')
            c.db.execute("UPDATE watch_outbox SET state='cancelled' WHERE key LIKE ? AND state='queued'", (f"failure:{old['id']}:%",))
        c.db.execute("INSERT INTO operations(id,user_id,action,case_id,command,random_id,state,created_at) VALUES (?,?,?,?,?,?,'sending',?)",
                     (op_id, user, action, case, text, secrets.randbits(63) or 1, now))
        if c.db.execute('SELECT 1 FROM watch_spy WHERE id=?', (case,)).fetchone():
            c.db.execute('INSERT INTO spy_operations VALUES (?,?)', (op_id, case))
        run['operation'] = op_id
        if run['status'] == 'requested':
            run['status'] = 'probe'
        c.ledger.put(KEY, run)
    return dict(c.db.execute('SELECT * FROM operations WHERE id=?', (op_id,)).fetchone())


async def tick(c, t):
    if state(c).get('status') not in ACTIVE:
        return
    try:
        await advance(c, t)
    except Flood:
        raise
    except Exception:
        pause(c, '無法核對案件、群組或回覆，請檢查服務後再用 /queue resume。')
        raise


async def advance(c, t):
    run = state(c)
    if not c.authorized(run['actor']) or not c.config.get('receiver_scope_confirmed'):
        pause(c, '啟動者的權限已撤銷，或接收端權限未確認。')
        return
    if run['operation']:
        op = dict(c.db.execute('SELECT * FROM operations WHERE id=?', (run['operation'],)).fetchone())
        if op['state'] not in ('acked', 'cancelled'):
            await t.reconcile_operation(c.ledger, op)
            op = dict(c.db.execute('SELECT * FROM operations WHERE id=?', (op['id'],)).fetchone())
        if op['state'] == 'acked':
            if run['status'] == 'probe':
                run['status'] = 'running'
                with c.db:
                    notice(c, run, 'passed', '✅ 重試成功，開始處理剩餘 +spb 寫入。')
            run['operation'] = None
        elif op['state'] == 'cancelled' and op['error'] == 'flood_wait':
            run['operation'] = None
        elif op['state'] == 'uncertain':
            pause(c, ('hexDrill 回覆操作失敗。' if rejected(op) else '未確認送出結果，不會重複發送。')
                  + f"\nsync:{uuid.UUID(op['id']).hex[:12]}")
            return
        elif op['state'] in ('sending', 'sent'):
            return
        else:
            pause(c, '操作狀態已變更，請先核對。')
            return
        with c.db:
            c.ledger.put(KEY, run)
        return
    waiting = pending(c)
    if any(not rejected(op) for op in waiting):
        pause(c, '仍有操作未確認結果，請先用 /sync 核對。')
        return
    todo = changes(c)
    change = tuple(run['probe']) if run['status'] in ('requested', 'probe') else (todo[0] if todo else None)
    if change is None:
        if waiting:
            pause(c, '仍有失敗案件不再符合收錄條件，請先核對。')
        else:
            run['status'] = 'done'
            with c.db:
                c.ledger.put(KEY, run)
                c.ledger.put('delivery_paused', False)
                notice(c, run, 'done', '✅ +spb 佇列已處理完成。')
        return
    if change not in todo:
        pause(c, '試送案件已撤銷或不再符合收錄條件，未重送。')
        return
    await t.check_destination()
    previous = [op for op in waiting if (op['action'], op['user_id']) == change[:2]]
    for op in previous:
        if not await t.confirmed_failure(op):
            pause(c, '無法核對原本的失敗回覆，未重送。')
            return
    # Recheck source and authorization after network awaits, immediately before claiming.
    if not c.authorized(run['actor']) or change not in changes(c):
        pause(c, '權限或來源案件已變更，未送出。')
        return
    operation = claim(c, run, change, previous)
    if operation is None:
        return
    try:
        message = await asyncio.wait_for(t.send(operation), timeout=25)
        c.ledger.sent(operation['id'], message)
    except Flood as error:
        c.ledger.flood(operation['id'], error.seconds)
        raise
    except BaseException:
        c.ledger.uncertain(operation['id'])
        pause(c, '發送結果不明，未重送。請先用 /sync 核對。')
        raise
