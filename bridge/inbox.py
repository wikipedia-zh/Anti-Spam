"""Private message intake; SPB alone scores and admits network bans."""
import time
import uuid

from core import PROTECTED

ROLE_REPLY = "這是 Spam Protection Bot 項目服務的角色帳號，不會回應一般訊息。如需協助，請加入討論群。\nhttps://t.me/SpamProtectionChat"


def initialize(control):
    control.db.executescript("""
        CREATE TABLE IF NOT EXISTS watch_inbox(id TEXT PRIMARY KEY,user_id INTEGER NOT NULL,message_id INTEGER NOT NULL,
            text TEXT NOT NULL,received_at INTEGER NOT NULL,status TEXT NOT NULL DEFAULT 'pending',
            score REAL,threshold REAL,processed_at INTEGER,notified INTEGER NOT NULL DEFAULT 0,
            UNIQUE(user_id,message_id));
        CREATE TABLE IF NOT EXISTS watch_greetings(user_id INTEGER PRIMARY KEY,sent_day INTEGER NOT NULL);
    """)
    with control.db:
        if control.config.get("dm_enabled") and control.ledger.get("dm_since") is None:
            control.ledger.put("dm_since", int(time.time()))


def capture(control, user, message_id, text, date):
    if control.config.get("owner_only_private") and user != control.owner:
        return
    if (not control.config.get("dm_enabled") or user <= 0 or user >= 2**52 or user in PROTECTED
            or user in (control.config["account_id"], control.config["hex_bot_id"])):
        return
    if date < control.ledger.get("dm_since", int(time.time())):
        return
    parts = text.strip().split()
    if parts and parts[0].split("@")[0].lower() in ("/sudo", "/unsudo", "/sync", "/spb_sync", "/evidence"):
        return
    now = int(time.time())
    with control.db:
        if control.authorized(user):
            greet(control, user)
            return
        # Empty media, long messages and forwarded content never trigger automatic bans.
        if not text.strip() or len(text) > 4096:
            greet(control, user)
            return
        count, per_user = control.db.execute("SELECT count(*),coalesce(sum(user_id=?),0) FROM watch_inbox WHERE received_at>?", (user, now - 86400)).fetchone()
        if count >= 200 or per_user >= 20:
            control.enqueue(f"inbox-limit:{now//86400}", control.owner, "⚠️ Spam Protection Bot 私訊檢測達到每日上限，新訊息暫不自動封禁，請檢查收件匣。")
            return
        control.db.execute("INSERT OR IGNORE INTO watch_inbox(id,user_id,message_id,text,received_at) VALUES (?,?,?,?,?)", (str(uuid.uuid4()), user, message_id, text, date))


def greet(control, user):
    day = (int(time.time()) + 8 * 3600) // 86400
    prior = control.db.execute("SELECT sent_day FROM watch_greetings WHERE user_id=?", (user,)).fetchone()
    if prior and prior[0] == day:
        return
    control.enqueue(f"greeting:{user}:{day}", user, ROLE_REPLY)
    control.db.execute("INSERT INTO watch_greetings VALUES (?,?) ON CONFLICT(user_id) DO UPDATE SET sent_day=excluded.sent_day", (user, day))


def outcomes(control):
    with control.db:
        pending = control.db.execute("SELECT id FROM watch_inbox WHERE status='pending' AND received_at<? LIMIT 1", (int(time.time()) - 120,)).fetchone()
        if pending:
            control.enqueue(f"inbox-pending:{int(time.time())//86400}", control.owner, "⚠️ Spam Protection Bot 私訊檢測超過兩分鐘未完成，請檢查後臺；這些訊息尚未自動封禁。")
        for row in control.db.execute("SELECT * FROM watch_inbox WHERE status!='pending' AND notified=0 LIMIT 20").fetchall():
            if row["status"] in ("ham", "exempt"):
                greet(control, row["user_id"])
            else:
                title = "已加入聯防黑名單" if row["status"] == "banned" else "待主持人審核"
                text = f"Spam Protection Bot 私訊檢測：{title}\n使用者：{row['user_id']}\n案件：#{row['id'].replace('-', '')[:8]}\n模型：{row['score']:.4f}；全域門檻：{row['threshold']:.4f}\n原文請在這裡用 /evidence #{row['id'].replace('-', '')[:8]} 查閱。"
                control.enqueue(f"inbox:{row['id']}", control.owner, text)
            control.db.execute("UPDATE watch_inbox SET notified=1 WHERE id=?", (row["id"],))
        control.db.execute("UPDATE watch_inbox SET text='' WHERE status IN ('ham','exempt') AND text!='' AND received_at<?", (int(time.time()) - 86400,))


def evidence(control, reference):
    import re
    import sqlite3
    from pathlib import Path
    value = reference.removeprefix("#").replace("-", "").lower()
    if not re.fullmatch(r"[0-9a-f]{8,32}", value):
        return "用法：/evidence #案件碼（僅主持人私訊可用）"
    rows = control.db.execute("SELECT id,user_id,text,status FROM watch_inbox WHERE replace(id,'-','') LIKE ? LIMIT 2", (value + "%",)).fetchall()
    if not rows:
        return "找不到私訊紀錄。"
    if len(rows) != 1:
        return "短碼對應多筆紀錄，請使用完整編號。"
    row = rows[0]
    status = row['status']
    source = sqlite3.connect(Path(control.config['source_db']).resolve(strict=True).as_uri() + '?mode=ro', uri=True)
    try:
        source.execute('PRAGMA query_only=ON')
        source.execute('PRAGMA trusted_schema=OFF')
        case = source.execute('SELECT status FROM cases WHERE id=?', (row['id'],)).fetchone()
        if case:
            status = case[0]
    finally:
        source.close()
    suffix = "\n（原文較長，只顯示前 2800 字）" if len(row["text"]) > 2800 else ""
    return f"案件：{row['id']}\n使用者：{row['user_id']}\n狀態：{status}\n以下是對方傳來的原文：\n\n" + (row["text"][:2800] or "（已清除）") + suffix
