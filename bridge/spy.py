"""Passive group intake and explicit group controls."""
import re
import time
import uuid

from core import PROTECTED
from telegram_client import Flood


def initialize(c):
    c.db.executescript("""
        CREATE TABLE IF NOT EXISTS spy_ignored(chat_id INTEGER PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS spy_groups(chat_id INTEGER PRIMARY KEY,joined_by INTEGER NOT NULL,created_at INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS watch_spy(id TEXT PRIMARY KEY,chat_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,user_id INTEGER NOT NULL,text TEXT NOT NULL,
            received_at INTEGER NOT NULL,status TEXT NOT NULL DEFAULT 'pending',
            score REAL,threshold REAL,UNIQUE(chat_id,message_id));
        CREATE INDEX IF NOT EXISTS watch_spy_pending ON watch_spy(status,received_at);
        CREATE TABLE IF NOT EXISTS watch_join_jobs(chat_id INTEGER,command_id INTEGER,actor INTEGER,
            target TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'queued',created_at INTEGER NOT NULL,next_at INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(chat_id,command_id));
        CREATE TABLE IF NOT EXISTS spy_operations(operation_id TEXT PRIMARY KEY,case_id TEXT NOT NULL);
    """)
    if c.config.get('spy_enabled'):
        settings = {'destination': c.config['destination_chat_id'],
                    'excluded': sorted(c.config.get('spy_excluded_chats', [])),
                    'bot_id': c.config['spb_bot_id']}
        pinned = c.ledger.get('spy_settings')
        if pinned is not None and pinned != settings:
            from core import StopBridge
            raise StopBridge('Spy destination or exclusions changed; inspect the pinned settings')
        with c.db:
            c.ledger.put('spy_settings', settings)


def excluded(c, chat):
    return (chat >= 0 or chat == c.config['destination_chat_id']
            or chat in c.config.get('spy_excluded_chats', [])
            or c.db.execute('SELECT 1 FROM spy_groups WHERE chat_id=?', (chat,)).fetchone() is None
            or c.db.execute('SELECT 1 FROM spy_ignored WHERE chat_id=?', (chat,)).fetchone() is not None)


def capture(c, chat, message, actor, text, date):
    now = int(time.time())
    if (not c.config.get('spy_enabled') or excluded(c, chat)
            or not isinstance(actor, int) or not 0 < actor < 2**52
            or actor in PROTECTED | {c.config['account_id'], c.config['hex_bot_id'], c.config.get('spb_bot_id')}
            or c.authorized(actor) or not text.strip() or not 0 <= now - date <= 120):
        return
    enrolled = c.db.execute('SELECT created_at FROM spy_groups WHERE chat_id=?', (chat,)).fetchone()
    if not enrolled or date < enrolled[0]:
        return
    with c.db:
        # Bound disk/model work during a raid, and do not keep ordinary chat history.
        c.db.execute("DELETE FROM watch_spy WHERE received_at<? AND status IN ('ham','exempt')", (now-86400,))
        c.db.execute("UPDATE watch_spy SET text='' WHERE received_at<? AND status!='pending'", (now-30*86400,))
        if c.db.execute("SELECT count(*) FROM watch_spy WHERE status='pending'").fetchone()[0] >= 500:
            c.enqueue('spy-backlog', c.owner, '⚠️ WATCH 掃描佇列已滿，部分新訊息未納入。')
            return
        c.db.execute('INSERT OR IGNORE INTO watch_spy(id,chat_id,message_id,user_id,text,received_at) VALUES (?,?,?,?,?,?)',
                     (str(uuid.uuid4()), chat, message, actor, text[:4096], now))


def target(value):
    match = re.fullmatch(r'(?:https://t\.me/\+|https://t\.me/joinchat/)([A-Za-z0-9_-]{8,128})', value)
    if match:
        return ('invite', match[1])
    if re.fullmatch(r'@[A-Za-z][A-Za-z0-9_]{3,31}', value):
        return ('username', value[1:])
    raise ValueError('請使用 /join @群組名稱 或 /join https://t.me/+邀請碼。')


def command(c, chat, message, actor, parts):
    if not c.config.get('spy_enabled'):
        return '群組掃描尚未啟用。'
    if parts[0] == '/join':
        if parts == ['/join','list']:
            groups=[r[0] for r in c.db.execute('SELECT chat_id FROM spy_groups ORDER BY chat_id')]
            return '已登記群組：\n' + ('\n'.join(f"{g} · {'不掃描' if excluded(c,g) else '掃描中'}" for g in groups) or '無')
        try:
            if len(parts) != 2:
                raise ValueError('請使用 /join @群組名稱 或 /join https://t.me/+邀請碼。')
            target(parts[1])
        except ValueError as e:
            return str(e)
        c.db.execute('INSERT OR IGNORE INTO watch_join_jobs(chat_id,command_id,actor,target,created_at) VALUES (?,?,?,?,?)',
                     (chat, message, actor, parts[1], int(time.time())))
        return None
    if actor != c.owner:
        return '只有主持人可以變更掃描範圍。'
    if parts == ['/ignore', 'list']:
        fixed = {c.config['destination_chat_id']} | set(c.config.get('spy_excluded_chats', []))
        extra = {r[0] for r in c.db.execute('SELECT chat_id FROM spy_ignored')}
        return '不掃描：\n' + '\n'.join(str(i) for i in sorted(fixed | extra))
    remove = len(parts) == 3 and parts[1] == 'del'
    if (len(parts) != 2 and not remove) or not re.fullmatch(r'-[1-9][0-9]{0,15}', parts[-1]):
        return '/ignore 群組ID：停止掃描\n/ignore del 群組ID：恢復掃描\n/ignore list：查看排除名單'
    group = int(parts[-1])
    if remove:
        if group == c.config['destination_chat_id'] or group in c.config.get('spy_excluded_chats', []):
            return '這個群組固定不掃描。'
        if not c.db.execute('SELECT 1 FROM spy_groups WHERE chat_id=?', (group,)).fetchone():
            return '這個群組尚未透過 /join 登記，不會掃描。'
        c.db.execute('DELETE FROM spy_ignored WHERE chat_id=?', (group,))
        return f'已恢復掃描 {group} 的新訊息。'
    c.db.execute('INSERT OR IGNORE INTO spy_ignored VALUES (?)', (group,))
    c.db.execute("UPDATE watch_spy SET status='exempt',text='' WHERE chat_id=? AND status='pending'", (group,))
    return f'已停止掃描 {group}。已送審的案件保留。'


async def joins(c, transport):
    for row in c.db.execute("SELECT * FROM watch_join_jobs WHERE state IN ('queued','waiting') AND next_at<=? ORDER BY created_at LIMIT 1",(int(time.time()),)).fetchall():
        key = (row['chat_id'], row['command_id'])
        with c.db:
            if not c.authorized(row['actor']) or int(time.time()) - row['created_at'] > (7*86400 if row['state']=='waiting' else 120):
                c.db.execute("UPDATE watch_join_jobs SET state='cancelled',target='' WHERE chat_id=? AND command_id=?", key)
                continue
            c.db.execute("UPDATE watch_join_jobs SET state='sending' WHERE chat_id=? AND command_id=?", key)
        try:
            if row['state']=='waiting':
                group = await transport.joined_group(row['target'],row['actor'])
                if group is None:
                    with c.db:
                        c.db.execute("UPDATE watch_join_jobs SET state='waiting',next_at=? WHERE chat_id=? AND command_id=?",(int(time.time())+300,*key))
                    continue
                pending = False
            else:
                group, pending = await transport.join_group(row['target'], row['actor'])
            if not pending:
                with c.db:
                    if not c.authorized(row['actor']):
                        raise PermissionError('Authorization revoked')
                    c.db.execute('INSERT OR IGNORE INTO spy_groups VALUES (?,?,?)',(group,row['actor'],int(time.time())))
            body = f'已送出入群申請：{group}。' if pending else f'已加入 {group}。' + ('此群組不掃描。' if excluded(c, group) else '只掃描新訊息，命中後送審。')
            state = 'waiting' if pending else 'done'
        except Flood:
            with c.db:
                c.db.execute("UPDATE watch_join_jobs SET state=? WHERE chat_id=? AND command_id=?", (row['state'],*key))
            raise
        except Exception as e:
            body, state = f'加入群組未確認成功（{type(e).__name__}），沒有自動重試。', 'uncertain'
        with c.db:
            c.db.execute('UPDATE watch_join_jobs SET state=?,target=?,next_at=? WHERE chat_id=? AND command_id=?', (state,row['target'] if state=='waiting' else '',int(time.time())+300,*key))
            c.enqueue(f'groupctl:{key[0]}:{key[1]}:{state}', key[0], body, row['actor'])


def interrupted(c):
    with c.db:
        for row in c.db.execute("SELECT * FROM watch_join_jobs WHERE state='sending'").fetchall():
            c.enqueue(f"groupctl:{row['chat_id']}:{row['command_id']}", row['chat_id'],
                      '入群操作因服務重啟中斷，請核對成員資格；沒有自動重試。', row['actor'])
        c.db.execute("UPDATE watch_join_jobs SET state='uncertain',target='' WHERE state='sending'")
