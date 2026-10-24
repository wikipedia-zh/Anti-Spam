"""Explicit review of forwarded SPB action logs; source database stays read-only."""
from datetime import datetime
from pathlib import Path
import re
import sqlite3
import time
import uuid

from core import PROTECTED, StopBridge, case_uuid


def initialize(c):
    c.db.executescript(Path(__file__).with_suffix('.sql').read_text(encoding='utf-8'))


def protected(c):
    return c.users() | {c.config['account_id'], c.config['hex_bot_id'], c.config.get('spb_bot_id')}


def row_item(row):
    return {'prefix': row['case_id'].replace('-', ''), 'user_id': row['user_id'],
            'chat_id': row['source_chat'], 'title': row['title']}


def capture(c, chat, message, actor, text, date):
    """Only the owner can submit a log. This never approves or sends a ban."""
    if chat != c.owner or actor != c.owner:
        return False
    item = parse(text)
    if item is None:
        return False
    now = int(time.time())
    with c.db:
        cursor = c.cursor(chat)
        if cursor is not None and message <= cursor:
            return True
        c.advance(chat, message)
        if date is None or not 0 <= now - date <= 86400:
            return True
        try:
            db = source(c.config['source_db'])
            try:
                case = validate(db, item, protected(c))
            finally:
                db.close()
            if c.db.execute('SELECT 1 FROM owned WHERE user_id=? AND active=1', (item['user_id'],)).fetchone():
                body = f"{item['user_id']} 已有 hexDrill +spb 成功回覆，未重複推送。"
            else:
                prior = c.db.execute('SELECT * FROM watch_log_requests WHERE case_id=?', (case['id'],)).fetchone()
                if prior and prior['status'] == 'approved':
                    body = '這個案件已確認，正在等待佇列推送或核對回覆；不會重複建立。'
                else:
                    if prior and (prior['status'] in ('rejected', 'invalid')
                                  or prior['review_state'] == 'uncertain'
                                  or now - prior['created_at'] > 86400):
                        # A new explicit forward retries the card, not a Hexdrill command.
                        c.db.execute("""UPDATE watch_log_requests SET status='pending',approved_by=NULL,
                            review_state='queued',review_chat=NULL,review_message=NULL,review_error=NULL,
                            submitted_message=?,created_at=?,updated_at=? WHERE id=? AND review_state!='sending'""",
                                     (message, now, now, prior['id']))
                    elif not prior:
                        c.db.execute("""INSERT INTO watch_log_requests
                            (id,case_id,user_id,source_chat,title,submitted_by,submitted_message,created_at,updated_at)
                            VALUES (?,?,?,?,?,?,?,?,?)""",
                                     (str(uuid.uuid4()), case['id'], item['user_id'], item['chat_id'], item['title'], actor, message, now, now))
                    body = (f"已讀取 {item['title']}\n對象：{item['user_id']}\n案例：#{item['prefix'][:8]}\n"
                            '請到報告群按「確認加入 +spb」或「取消」。確認前不會因這次轉發推送。')
        except ValueError as error:
            body = str(error)
        except Exception:
            body = '暫時無法核對 SPB 案件，未建立確認請求；請稍後重新轉發。'
        c.enqueue(f'log:{chat}:{message}', chat, body, actor)
    return True


def approved_cases(c):
    rows = c.db.execute("SELECT * FROM watch_log_requests WHERE status='approved' AND approved_by=?", (c.owner,)).fetchall()
    if not rows:
        return {}
    db = source(c.config['source_db'])
    cases = {}
    try:
        for row in rows:
            try:
                case = validate(db, row_item(row), protected(c))
            except ValueError:
                continue
            cases[case['id']] = {'case_id': case['id'], 'user_id': case['target_user_id'],
                                  'reason': 'manual_log', 'created_at': case['created_at'],
                                  'display_score': None, 'log_message_id': case['log_message_id']}
    finally:
        db.close()
    return cases


def augment(c, snapshot):
    manual = approved_cases(c)
    return dict(snapshot, cases={**snapshot['cases'], **manual})


def outcomes(c):
    with c.db:
        for row in c.db.execute("SELECT * FROM watch_log_requests WHERE status='invalid'").fetchall():
            c.enqueue('log-invalid:' + row['id'] + ':' + str(row['created_at']), c.owner,
                      f"案件 #{row['case_id'].replace('-', '')[:8]} 已失效、受保護或確認卡已過期，未推送。需要再處理時請重新轉發 log。")
        for row in c.db.execute("SELECT * FROM watch_log_requests WHERE review_state='uncertain' AND status='pending'").fetchall():
            c.enqueue('log-card-error:' + row['id'] + ':' + str(row['created_at']), c.owner,
                      f"案件 #{row['case_id'].replace('-', '')[:8]} 的確認卡未能確認送達；尚未批准推送。請重新轉發該 log 以重試確認卡。")


TITLES = {
    '訪客模式機器人封禁': ('guest_bot_ban', {'guest_bot_banned'}),
    '自動封禁': ('auto_ban', {'auto_banned', 'banned_delete_failed'}),
    '受理封禁': ('report_approved', {'approved_and_banned', 'banned_delete_failed'}),
    '逕行封禁': ('spam_ban', {'force_approved'}),
}
HEADER = re.compile(
    r'\A\s*(?:\[[^\]\n]{1,80}\]\s*)?(?:Spam Protection Log\s*[:：]\s*)?'
    r'(?P<title>' + '|'.join(TITLES) + r')\s+'
    r'對象\s*[:：]\s*(?P<user>[1-9][0-9]{0,15})\s+(?P<name>.*?)\s+'
    r'群組\s*[:：]\s*(?P<chat>-[1-9][0-9]{0,15})\s+'
    r'案例\s*[:：]\s*#(?P<case>[0-9a-fA-F]{8}(?:[0-9a-fA-F-]{0,28}))(?=\s)'
    r'(?P<details>.*?)\s+(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC\+8(?=\s|$)',
    re.DOTALL,
)


def parse(text):
    if not isinstance(text, str) or len(text) > 16384:
        return None
    match = HEADER.match(text)
    if not match:
        return None
    item = match.groupdict()
    # Only the action header is interpreted. Spam evidence is never a command.
    if re.search(r'(?:對象|群組|案例)\s*[:：]', item['name'] + item['details']):
        return None
    prefix = item['case'].replace('-', '').lower()
    if len(prefix) not in (8, 32) or (len(item['case']) != 8 and not case_uuid(item['case'].lower())):
        return None
    try:
        datetime.strptime(item['date'], '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return None
    user, chat = int(item['user']), int(item['chat'])
    if not 0 < user < 2**52 or not -(2**52) < chat < 0:
        return None
    return {'prefix': prefix, 'user_id': user, 'chat_id': chat, 'title': item['title']}


def source(path):
    db = sqlite3.connect(Path(path).resolve(strict=True).as_uri() + '?mode=ro', uri=True, timeout=3)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    db.execute('PRAGMA trusted_schema=OFF')
    db.execute('BEGIN')
    if db.execute('PRAGMA user_version').fetchone()[0] not in (40, 41):
        db.close()
        raise StopBridge('Unsupported SPB source schema')
    return db


def validate(db, item, protected):
    rows = db.execute("""SELECT c.*,j.ban_done,j.outcome_unknown
        FROM cases c LEFT JOIN origin_ban_jobs j ON j.case_id=c.id
        WHERE replace(c.id,'-','') LIKE ? LIMIT 2""", (item['prefix'] + '%',)).fetchall()
    if len(rows) != 1:
        raise ValueError('案件碼不存在或對應多筆紀錄，請轉發完整的原始 log。')
    case = dict(rows[0])
    action, statuses = TITLES[item['title']]
    if (not case_uuid(case['id']) or case['target_user_id'] != item['user_id']
            or case['chat_id'] != item['chat_id'] or case['action'] != action):
        raise ValueError('log 的對象、群組或封禁類型與 SPB 案件不符，未建立確認請求。')
    if (case['status'] not in statuses or case['netban_eligible'] != 1
            or (case['ban_done'] is not None and (case['ban_done'] != 1 or case['outcome_unknown'] != 0))):
        raise ValueError('案件已撤銷、尚未確認封禁，或不符合收錄條件，未推送。')
    excluded = PROTECTED | set(protected)
    for table in ('global_whitelist', 'maintainers', 'reviewers'):
        excluded.update(row[0] for row in db.execute(f'SELECT user_id FROM {table}'))
    if case['target_user_id'] in excluded:
        raise ValueError('此帳號目前受保護或在白名單內，未推送。')
    return case
