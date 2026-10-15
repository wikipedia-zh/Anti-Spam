"""Authorized deletion of a fixed message range."""
import time

from core import StopBridge
from telegram_client import Flood


def initialize(control):
    control.db.execute("""CREATE TABLE IF NOT EXISTS purge_jobs(
        chat_id INTEGER NOT NULL,command_id INTEGER NOT NULL,actor INTEGER NOT NULL,
        state TEXT NOT NULL DEFAULT 'queued',start_id INTEGER,end_id INTEGER,
        deleted INTEGER NOT NULL DEFAULT 0,error TEXT,created_at INTEGER NOT NULL,
        PRIMARY KEY(chat_id,command_id))""")
    control.db.commit()


def interrupted(control):
    with control.db:
        rows = control.db.execute("SELECT * FROM purge_jobs WHERE state IN ('queued','running')").fetchall()
        for row in rows:
            control.db.execute("UPDATE purge_jobs SET state='stopped',error='restart' WHERE chat_id=? AND command_id=?",
                               (row['chat_id'], row['command_id']))
            control.enqueue(f"purge:{row['chat_id']}:{row['command_id']}", control.owner,
                f"⚠️ 清除訊息因服務重啟而停止，已確認刪除 {row['deleted']} 則；未確認的部分不會自動重試。")


def request(control, chat, message_id, actor):
    if not control.authorized(actor):
        return "只有主持人及授權人員可以使用 /purge。"
    if control.db.execute("SELECT 1 FROM purge_jobs WHERE state IN ('queued','running') LIMIT 1").fetchone():
        return "上一項清除工作尚未完成。"
    control.db.execute("INSERT OR IGNORE INTO purge_jobs(chat_id,command_id,actor,created_at) VALUES (?,?,?,?)",
                       (chat, message_id, actor, int(time.time())))
    return None


async def run(control, transport, peers):
    row = control.db.execute("SELECT * FROM purge_jobs WHERE state='queued' ORDER BY created_at LIMIT 1").fetchone()
    if row is None or row['chat_id'] not in peers:
        return
    chat, command = row['chat_id'], row['command_id']
    with control.db:
        claimed = control.db.execute("UPDATE purge_jobs SET state='running' WHERE chat_id=? AND command_id=? AND state='queued'",
                                     (chat, command)).rowcount
    if not claimed:
        return
    peer, deleted = peers[chat], 0
    try:
        if not control.authorized(row['actor']) or not (chat < 0 or chat == row['actor']):
            raise StopBridge("清除權限已撤銷。")
        start, end = await transport.purge_bounds(peer, chat, command, row['actor'])
        with control.db:
            control.db.execute("UPDATE purge_jobs SET start_id=?,end_id=? WHERE chat_id=? AND command_id=?",
                               (start, end, chat, command))
        cursor = start - 1
        while True:
            ids = await transport.purge_batch(peer, chat, cursor, end)
            if not ids:
                break
            if len(ids) > 100 or any(not cursor < value <= end for value in ids):
                raise StopBridge("Invalid purge batch")
            if not control.authorized(row['actor']):
                raise StopBridge("清除權限已撤銷。")
            await transport.delete_purge_batch(peer, chat, ids, row['actor'])
            deleted += len(ids)
            cursor = max(ids)
            with control.db:
                control.db.execute("UPDATE purge_jobs SET deleted=? WHERE chat_id=? AND command_id=?", (deleted, chat, command))
        state, error, body = 'done', None, f"已刪除 {deleted} 則訊息。"
    except Exception as exc:
        state, error = 'stopped', type(exc).__name__
        if isinstance(exc, StopBridge):
            reason = str(exc)
        elif isinstance(exc, Flood):
            reason = "Telegram 要求暫停，請稍後再操作。"
        else:
            reason = "Telegram 未確認刪除成功，請檢查刪除權限。"
        body = f"⚠️ 清除已停止，已確認刪除 {deleted} 則。\n{reason}"
    with control.db:
        control.db.execute("UPDATE purge_jobs SET state=?,error=? WHERE chat_id=? AND command_id=?", (state, error, chat, command))
        control.enqueue(f"purge:{chat}:{command}", chat, body, row['actor'])
