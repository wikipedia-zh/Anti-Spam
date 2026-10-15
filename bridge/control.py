"""Owner-controlled queries and delivery alerts. Never creates moderation operations."""
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
import re
import secrets
import sqlite3
import time

from core import StopBridge
from telegram_client import Flood

HELP = "SPB-WATCH\n/sync sync:追蹤碼\n/sync #案件碼\n/sync 使用者ID\n群組請用 /spb_sync，避免與 SPB 重複回覆。"
OWNER_HELP = "\n/sudo ID：授權查詢\n/sudo del ID：撤權\n/sudo list：查看授權"
STATES = {"sending": "送出結果待確認", "sent": "已送出，等候 hexDrill 確認",
          "uncertain": "未確認成功，已停止後續封禁", "acked": "hexDrill 已確認成功",
          "cancelled": "已取消這次嘗試", "rejected": "已拒絕"}
REASONS = {"model_display_1": "模型顯示 1.00", "regex": "REGEX 命中",
           "approved_report": "項目人員核准舉報", "approved_report_legacy": "舊舉報（核准紀錄待核對）"}


def lookup(conn, value):
    value = value.strip().lower()
    if value.startswith("sync:"):
        column, prefix = "id", value[5:].replace("-", "")
        minimum = 12
    elif value.startswith("#"):
        column, prefix, minimum = "case_id", value[1:].replace("-", ""), 8
    elif re.fullmatch(r"[1-9][0-9]{0,15}", value) and int(value) < 2**52:
        rows = conn.execute("SELECT body,updated_at FROM operation_reports WHERE user_id=? ORDER BY created_at DESC,id DESC LIMIT 3", (int(value),)).fetchall()
        return rows
    else:
        column, prefix, minimum = "id", value.replace("-", ""), 12
    if not re.fullmatch(r"[0-9a-f]{%d,32}" % minimum, prefix):
        raise ValueError("請輸入 sync:追蹤碼、#案件碼或使用者 ID。")
    ids = conn.execute(f"SELECT DISTINCT {column} FROM operation_reports WHERE replace({column},'-','') LIKE ? LIMIT 2", (prefix + "%",)).fetchall()
    if len(ids) > 1:
        raise ValueError("短碼對應多筆紀錄，請使用完整編號。")
    if not ids:
        return []
    return conn.execute(f"SELECT body,updated_at FROM operation_reports WHERE {column}=? ORDER BY created_at DESC,id DESC LIMIT 3", (ids[0][0],)).fetchall()


class Control:
    def __init__(self, config, ledger):
        self.config, self.ledger, self.db = config, ledger, ledger.conn
        self.owner = config.get("operator_id")
        if not self.owner:
            raise StopBridge("Set operator_id before starting watch")
        pinned = ledger.get("watch_owner")
        if pinned is not None and pinned != self.owner:
            raise StopBridge("Watch owner changed; inspect the pinned identity")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS sudo_users(user_id INTEGER PRIMARY KEY,granted_at INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS control_cursors(chat_id INTEGER PRIMARY KEY,message_id INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS control_audit(chat_id INTEGER,message_id INTEGER,actor INTEGER,command TEXT,created_at INTEGER,PRIMARY KEY(chat_id,message_id));
            CREATE TABLE IF NOT EXISTS watch_outbox(key TEXT PRIMARY KEY,chat_id INTEGER NOT NULL,actor INTEGER,
                body TEXT NOT NULL,random_id INTEGER NOT NULL,state TEXT NOT NULL DEFAULT 'queued',
                created_at INTEGER NOT NULL,message_id INTEGER,error TEXT,next_at INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS operation_reports(id TEXT PRIMARY KEY,user_id INTEGER NOT NULL,case_id TEXT NOT NULL,
                body TEXT NOT NULL,created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL);
        """)
        with self.db:
            ledger.put("watch_owner", self.owner)
        from inbox import initialize
        initialize(self)

    def authorized(self, user):
        return user == self.owner or self.db.execute("SELECT 1 FROM sudo_users WHERE user_id=?", (user,)).fetchone() is not None

    def users(self):
        return {self.owner} | {r[0] for r in self.db.execute("SELECT user_id FROM sudo_users")}

    def enqueue(self, key, chat, body, actor=None):
        self.db.execute("INSERT OR IGNORE INTO watch_outbox(key,chat_id,actor,body,random_id,created_at) VALUES (?,?,?,?,?,?)",
                        (key, chat, actor, body[:3500], secrets.randbits(63) or 1, int(time.time())))

    def cursor(self, chat):
        row = self.db.execute("SELECT message_id FROM control_cursors WHERE chat_id=?", (chat,)).fetchone()
        return row[0] if row else None

    def advance(self, chat, message):
        self.db.execute("INSERT INTO control_cursors VALUES (?,?) ON CONFLICT(chat_id) DO UPDATE SET message_id=max(message_id,excluded.message_id)", (chat, message))

    def handle(self, chat, message_id, actor, text, trusted=True):
        with self.db:
            cursor = self.cursor(chat)
            if cursor is None or message_id <= cursor:
                return
            self.advance(chat, message_id)
            if (not trusted or not self.authorized(actor)
                    or (chat > 0 and self.config.get("owner_only_private") and actor != self.owner)
                    or (chat != actor and chat != self.config["destination_chat_id"])):
                return
            parts = text.strip().split()
            if not parts or len(text) > 160:
                return
            command = parts[0].lower()
            if command not in ("/sync", "/spb_sync", "/sudo", "/unsudo", "/help", "/start", "/evidence"):
                return
            if chat < 0 and command in ("/sync", "/help", "/start"):
                return
            # A small per-actor limit also bounds replies after a large backlog.
            now = int(time.time())
            count = self.db.execute("SELECT count(*) FROM control_audit WHERE actor=? AND created_at>?", (actor, now - 60)).fetchone()[0]
            if count >= 6:
                return
            self.db.execute("INSERT INTO control_audit VALUES (?,?,?,?,?)", (chat, message_id, actor, command, now))
            if command == "/evidence":
                if actor != self.owner or chat != actor:
                    return
                from inbox import evidence
                body = evidence(self, parts[1] if len(parts) == 2 else "")
            elif command in ("/sudo", "/unsudo"):
                if actor != self.owner:
                    body = "只有主持人可以授權或撤權。"
                else:
                    body = self.sudo(parts)
            elif command in ("/help", "/start") or len(parts) != 2:
                body = HELP + (OWNER_HELP if actor == self.owner else "")
            else:
                try:
                    rows = lookup(self.db, parts[1])
                    body = "\n\n".join(row[0] for row in rows) or "找不到同步紀錄；不代表此人沒有 SPB 案件。"
                    if rows and any(now - row[1] > 90 for row in rows):
                        body = "⚠️ 資料未能更新，以下為上次讀取結果。\n" + body
                except ValueError as error:
                    body = str(error)
            self.enqueue(f"reply:{chat}:{message_id}", chat, body, actor)

    def sudo(self, parts):
        if parts == ["/sudo", "list"]:
            return "主持人：%s\n可查詢：%s" % (self.owner, "、".join(str(i) for i in sorted(self.users() - {self.owner})) or "無")
        remove = parts[0] == "/unsudo" or (len(parts) == 3 and parts[1] == "del")
        valid = len(parts) == 2 or (len(parts) == 3 and parts[1] in ("add", "del"))
        if not valid or not re.fullmatch(r"[1-9][0-9]{0,15}", parts[-1]) or int(parts[-1]) >= 2**52:
            return OWNER_HELP.strip()
        user = int(parts[-1])
        if user in (self.owner, self.config["account_id"], self.config["hex_bot_id"]):
            return "此帳號不能變更授權。"
        if remove:
            self.db.execute("DELETE FROM sudo_users WHERE user_id=?", (user,))
            self.db.execute("UPDATE watch_outbox SET state='cancelled' WHERE actor=? AND state='queued'", (user,))
            self.db.execute("DELETE FROM control_cursors WHERE chat_id=?", (user,))
            return f"已撤銷 {user} 的查詢權限。"
        if len(self.users()) >= 21 and not self.authorized(user):
            return "最多授權 20 人，請先移除不再需要的帳號。"
        self.db.execute("INSERT OR IGNORE INTO sudo_users VALUES (?,?)", (user, int(time.time())))
        where = "請對方在指定群組使用 /spb_sync" if self.config.get("owner_only_private") else "請對方私訊 SPB-WATCH /help"
        return f"已授權 {user} 查詢。{where}；不能授權他人或操作封禁。"

    def refresh(self):
        # Read all related cases in one source transaction; never open it for writes.
        source = sqlite3.connect(Path(self.config["source_db"]).resolve(strict=True).as_uri() + "?mode=ro", uri=True, timeout=3)
        source.row_factory = sqlite3.Row
        try:
            source.execute("PRAGMA query_only=ON")
            source.execute("PRAGMA trusted_schema=OFF")
            source.execute("BEGIN")
            operations = self.db.execute("SELECT * FROM operations ORDER BY created_at,id LIMIT 10001").fetchall()
            if len(operations) > 10000:
                raise StopBridge("Watch report limit reached")
            now = int(time.time())
            stamp = datetime.fromtimestamp(now, timezone(timedelta(hours=8))).strftime("%m-%d %H:%M HKT")
            reports = []
            for op in operations:
                cases = source.execute("SELECT id,action,status,model_score,matched_rule_pattern FROM cases WHERE target_user_id=? ORDER BY (id=?) DESC,created_at DESC LIMIT 5", (op["user_id"], op["case_id"])).fetchall()
                lines = [f"sync:{op['id'].replace('-', '')[:12]}", f"使用者：{op['user_id']}", "操作：" + ("加入 +spb" if op["action"] == "ban" else "移除 +spb"), "結果：" + STATES.get(op["state"], "未知狀態")]
                if op["error"]:
                    lines.append("錯誤：" + ("hexDrill 回覆操作失敗" if op["error"] == "receiver_failed" else "未確認／需人工核對"))
                for case in cases:
                    reason = "舉報核准" if case["action"] == "report_approved" else ("REGEX" if re.search(r"(?:^|；)REGEX@[1-9][0-9]*(?:；|$)", case["matched_rule_pattern"] or "") else "模型判定")
                    if isinstance(case["model_score"], (int, float)):
                        reason += f"（{case['model_score']:.4f}）"
                    lines.append(f"案件：#{case['id'].replace('-', '')[:8]} · {reason} · {case['status']}")
                if not cases:
                    lines.append("來源案件不存在，請核對 SPB。")
                if op["message_id"]:
                    group = str(self.config["destination_chat_id"])
                    if group.startswith("-100"):
                        lines.append(f"指令：https://t.me/c/{group[4:]}/{op['message_id']}")
                if op["reply_id"]:
                    lines.append(f"回覆 ID：{op['reply_id']}")
                lines.append("更新：" + stamp)
                reports.append((op["id"], op["user_id"], op["case_id"], "\n".join(lines)[:1050], op["created_at"], now))
            with self.db:
                self.db.executemany("INSERT INTO operation_reports VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body,updated_at=excluded.updated_at", reports)
                for op in operations:
                    if op["state"] == "acked":
                        self.db.execute("UPDATE watch_outbox SET state='cancelled' WHERE key LIKE ? AND state='queued'", (f"failure:{op['id']}:%",))
                    if op["state"] == "uncertain":
                        self.ledger.put("delivery_paused", True)
                        report = next(r[3] for r in reports if r[0] == op["id"])
                        for chat in (self.owner, self.config["destination_chat_id"]):
                            self.enqueue(f"failure:{op['id']}:{chat}", chat, "⚠️ SPB-WATCH 同步未成功，後續封禁已暫停。\n" + report)
        finally:
            source.close()


class Watch:
    def __init__(self, config, ledger, transport):
        self.control = Control(config, ledger)
        self.transport = transport
        self.incoming_peers = {}
        self.polled_at = {}
        self.group_checked_at = 0
        self.group_ok = False
        if hasattr(transport, "bind_ledger"):
            transport.bind_ledger(ledger)

    async def incoming(self, chat, message_id, actor, text, peer, date=None):
        c = self.control
        if chat > 0 and c.config.get("owner_only_private") and actor != c.owner:
            return
        if chat != actor and chat != c.config["destination_chat_id"]:
            return
        if not c.authorized(actor) and not (chat == actor and c.config.get("dm_enabled")):
            return
        if actor == c.owner:
            with c.db:
                c.ledger.put("watch_last_owner_event", {"message_id": message_id,
                    "received_at": int(time.time()), "message_at": date})
        self.incoming_peers.pop(chat, None)
        self.incoming_peers[chat] = peer
        while len(self.incoming_peers) > 512:
            del self.incoming_peers[next(iter(self.incoming_peers))]
        if hasattr(self.transport, "remember_peer"):
            self.transport.remember_peer(chat, peer)
        try:
            if chat < 0:
                await self.transport.check_destination()
            if c.cursor(chat) is None:
                with c.db:
                    c.advance(chat, message_id - 1)
            c.handle(chat, message_id, actor, text)
            if chat == actor:
                self.capture(actor, message_id, text, date or int(time.time()))
            await self.deliver({chat: peer}, key=f"reply:{chat}:{message_id}")
        except Flood:
            pass  # Leave the response queued until Telegram's cooldown expires.
        except Exception as error:
            print("watch command:", type(error).__name__, flush=True)

    def capture(self, actor, message_id, text, date):
        if self.control.authorized(actor) and text.strip().split(" ")[0] in ("/help", "/start"):
            return
        from inbox import capture
        capture(self.control, actor, message_id, text, date)

    async def tick(self):
        try:
            await self.check_and_poll()
        except Flood:
            pass  # Stop this entire RPC cycle, not just the current conversation.

    async def check_and_poll(self):
        c, t = self.control, self.transport
        if getattr(t, "wait_until", 0) > int(time.time()):
            return
        if int(time.time()) - self.group_checked_at >= 60:
            self.group_ok = False
            self.group_checked_at = int(time.time())
            try:
                await t.check_destination()
                await t.reconcile(c.ledger)
                self.group_ok = True
                with c.db:
                    c.ledger.put("watch_group_verified_at", int(time.time()))
            except Flood:
                raise
            except Exception as error:
                print("watch destination:", type(error).__name__, flush=True)
                with c.db:
                    c.enqueue("destination-unavailable", c.owner, "⚠️ SPB-WATCH 無法核對指定群組或封禁回覆。群組輸出已停止，請檢查成員及權限。")
        group_ok = self.group_ok
        try:
            c.refresh()
        except Exception as error:
            print("watch source:", type(error).__name__, flush=True)
            with c.db:
                c.enqueue("source-unavailable", c.owner, "⚠️ SPB-WATCH 無法更新 SPB 案件。查詢可能顯示舊資料，請檢查服務。")
        private_users = {c.owner} if c.config.get("owner_only_private") else c.users()
        peers = dict(self.incoming_peers)
        peers.update(await t.control_peers(private_users, group_ok))
        if not group_ok:
            peers.pop(c.config["destination_chat_id"], None)
        # Events handle new messages immediately. Poll at most one authorized
        # conversation per cycle to recover missed commands after reconnecting.
        candidates = private_users | ({c.config["destination_chat_id"]} if group_ok else set())
        for chat in sorted(candidates, key=lambda chat: (self.polled_at.get(chat, 0), chat)):
            if chat not in peers or int(time.time()) - self.polled_at.get(chat, 0) < 60:
                continue
            self.polled_at[chat] = int(time.time())
            try:
                await self.poll(chat, peers[chat])
            except Flood:
                raise
            except Exception as error:
                print("watch commands:", chat, type(error).__name__, flush=True)
            break
        from inbox import outcomes
        outcomes(c)
        await self.deliver(peers)

    async def poll(self, chat, peer):
        c, t = self.control, self.transport
        if chat > 0 and c.config.get("owner_only_private") and chat != c.owner:
            return
        cursor = c.cursor(chat)
        if cursor is None:
            grant = c.db.execute("SELECT granted_at FROM sudo_users WHERE user_id=?", (chat,)).fetchone()
            latest = await t.control_messages(peer, None if not grant else 0)
            with c.db:
                c.advance(chat, max((m["id"] for m in latest if not grant or m["date"] < grant[0]), default=0))
            if grant:
                for message in latest:
                    if message["date"] >= grant[0]:
                        c.handle(chat, message["id"], message["actor"], message["text"], message["trusted"])
            if chat > 0:
                for message in latest:
                    if message["trusted"] and message["actor"] == chat:
                        self.capture(chat, message["id"], message["text"], message["date"])
            return
        for message in await t.control_messages(peer, cursor):
            c.handle(chat, message["id"], message["actor"], message["text"], message["trusted"])
            if chat > 0 and message["trusted"] and message["actor"] == chat:
                self.capture(chat, message["id"], message["text"], message["date"])

    async def deliver(self, peers, key=None):
        c, t = self.control, self.transport
        rows = c.db.execute("SELECT * FROM watch_outbox WHERE state IN ('queued','sending') AND next_at<=? AND (? IS NULL OR key=?) ORDER BY created_at,key LIMIT 20", (int(time.time()), key, key)).fetchall()
        for row in rows:
            if (c.config.get("owner_only_private") and row["chat_id"] > 0
                    and row["chat_id"] != c.owner):
                with c.db:
                    c.db.execute("UPDATE watch_outbox SET state='cancelled' WHERE key=?", (row["key"],))
                continue
            if row["actor"] is not None and not c.authorized(row["actor"]):
                with c.db:
                    c.db.execute("UPDATE watch_outbox SET state='cancelled' WHERE key=?", (row["key"],))
                continue
            peer = peers.get(row["chat_id"])
            if peer is None:
                continue
            try:
                if row["chat_id"] < 0:
                    await t.check_destination()
                if row["state"] == "sending":
                    message_id = await t.recover_notice(peer, row)
                    if not message_id:
                        with c.db:
                            c.db.execute("UPDATE watch_outbox SET next_at=? WHERE key=?", (int(time.time()) + 300, row["key"]))
                        continue  # Unknown RPC outcome: never blindly repeat a notice.
                else:
                    with c.db:
                        claimed = c.db.execute("UPDATE watch_outbox SET state='sending' WHERE key=? AND state='queued'", (row["key"],)).rowcount
                    if not claimed:
                        continue
                    message_id = await t.send_notice(peer, row)
                with c.db:
                    c.db.execute("UPDATE watch_outbox SET state='sent',message_id=?,error=NULL WHERE key=?", (message_id, row["key"]))
                    if row["chat_id"] == c.owner and row["actor"] == c.owner:
                        c.ledger.put("watch_last_owner_reply", {"key": row["key"],
                            "sent_at": int(time.time()), "message_id": message_id})
            except Flood as error:
                with c.db:
                    c.db.execute("UPDATE watch_outbox SET state=?,next_at=?,error='flood_wait' WHERE key=?", (row["state"], int(time.time()) + error.seconds + 5, row["key"]))
                raise
            except Exception as error:
                with c.db:
                    c.db.execute("UPDATE watch_outbox SET error=?,next_at=? WHERE key=?", (type(error).__name__, int(time.time()) + 300, row["key"]))
                print("watch notice:", row["chat_id"], type(error).__name__, flush=True)

    async def run(self):
        self.transport.install_control_handler(self.incoming)
        while True:
            try:
                await self.tick()
            except Exception as error:
                print("watch:", type(error).__name__, flush=True)
            await asyncio.sleep(15)
