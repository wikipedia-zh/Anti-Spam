"""The only module allowed to send Telegram messages."""
import asyncio
from functools import wraps
import json
import os
from pathlib import Path
import time

from core import HEX_USERNAME, StopBridge, valid_ack


class Flood(Exception):
    def __init__(self, seconds):
        self.seconds = seconds


def flood_seconds(error):
    if isinstance(error, Flood):
        return error.seconds
    try:
        from telethon.errors import FloodWaitError
    except ImportError:
        return None
    return error.seconds if isinstance(error, FloodWaitError) else None


def guarded(method):
    @wraps(method)
    async def call(self, *args, **kwargs):
        remaining = self.wait_until - int(time.time())
        if remaining > 0:
            raise Flood(remaining)
        try:
            return await method(self, *args, **kwargs)
        except Exception as error:
            if isinstance(error, Flood) and self.wait_until > int(time.time()):
                raise
            seconds = flood_seconds(error)
            if seconds is None:
                raise
            self.wait_until = max(self.wait_until, int(time.time()) + seconds + 5)
            if self.ledger is not None:
                with self.ledger.conn:
                    self.ledger.put("telegram_wait_until", self.wait_until)
            print("Telegram rate limit; waiting until", self.wait_until, flush=True)
            raise Flood(self.wait_until - int(time.time())) from None
    return call


def private_file(path):
    path = Path(path).resolve(strict=True)
    if os.name == "posix" and path.stat().st_mode & 0o077:
        raise StopBridge("Credential files must have mode 600")
    return path


def make_client(config):
    from telethon import TelegramClient
    credentials = json.loads(private_file(config["credentials_file"]).read_text())
    session = Path(config["session_file"])
    if session.suffix != ".session":
        raise StopBridge("Use an explicit .session filename")
    if session.exists():
        private_file(session)
    client = TelegramClient(str(session), int(credentials["api_id"]), credentials["api_hash"],
        request_retries=0, connection_retries=2, timeout=10, flood_sleep_threshold=0,
        receive_updates=bool(config.get("operator_id")), device_model="SPB-WATCH", app_version="1.1")
    client.session.save_entities = False
    return client


class Telegram:
    def __init__(self, client, config):
        self.client, self.config = client, config
        self.destination = None
        self.ledger = None
        self.wait_until = 0
        self.peers = {}
        self.peers_checked_at = 0

    def bind_ledger(self, ledger):
        self.ledger = ledger
        self.wait_until = ledger.get("telegram_wait_until", 0)

    def remember_peer(self, chat, peer):
        if chat > 0:
            self.peers.pop(chat, None)
            self.peers[chat] = peer
            while len(self.peers) > 512:
                del self.peers[next(iter(self.peers))]

    @guarded
    async def check_destination(self):
        from telethon import utils
        me = await self.client.get_me()
        if not me or me.bot or me.id != self.config["account_id"]:
            raise StopBridge("Wrong Telegram account")
        hex_bot = await self.client.get_entity(HEX_USERNAME)
        if not hex_bot.bot or hex_bot.id != self.config["hex_bot_id"]:
            raise StopBridge("hexDrill identity changed")
        if self.destination is None:
            async for dialog in self.client.iter_dialogs():
                if dialog.id == self.config["destination_chat_id"]:
                    self.destination = dialog.entity
                    break
        if self.destination is None:
            raise StopBridge("Destination group not found")
        fresh = await self.client.get_entity(self.destination)
        if (utils.get_peer_id(fresh) != self.config["destination_chat_id"] or getattr(fresh, "username", None)
                or getattr(fresh, "broadcast", False) or getattr(fresh, "left", False)):
            raise StopBridge("Destination must remain the pinned private group")
        members = await self.client.get_participants(fresh, limit=10)
        ids = {m.id for m in members}
        allowed = {me.id, hex_bot.id} | set(self.config["allowed_observers"])
        if not {me.id, hex_bot.id} <= ids or not ids <= allowed or members.total != len(members):
            raise StopBridge("Destination membership changed or cannot be verified")
        self.destination = fresh

    @guarded
    async def send(self, operation):
        from telethon import errors, functions
        try:
            result = await self.client(functions.messages.SendMessageRequest(
                peer=await self.client.get_input_entity(self.destination), message=operation["command"],
                random_id=operation["random_id"], no_webpage=True))
        except errors.FloodWaitError as error:
            raise Flood(error.seconds) from None
        # An accepted RPC is not yet a successful hexDrill ban.
        if getattr(result, "id", None):
            return result.id
        for update in getattr(result, "updates", []):
            if getattr(update, "random_id", None) == operation["random_id"]:
                return update.id
        raise StopBridge("Telegram accepted the request without a message mapping")

    @guarded
    async def recover(self, operation):
        matches = []
        async for message in self.client.iter_messages(self.destination, limit=200):
            if (message.sender_id == self.config["account_id"] and message.out
                    and not message.fwd_from and message.raw_text == operation["command"]
                    and int(message.date.timestamp()) >= operation["created_at"] - 5):
                matches.append(message.id)
        if len(matches) > 1:
            raise StopBridge("Duplicate command messages; inspect the group")
        return matches[0] if matches else None

    @guarded
    async def acknowledgement(self, operation):
        # Only replies from the pinned bot to our exact command count.
        command = await self.client.get_messages(self.destination, ids=operation["message_id"])
        if not command or command.sender_id != self.config["account_id"] or not command.out or command.raw_text != operation["command"]:
            raise StopBridge("Sent command was removed or changed")
        async for message in self.client.iter_messages(self.destination, limit=200):
            if message.id <= operation["message_id"]:
                break
            candidate = {"chat_id": message.chat_id, "sender_id": message.sender_id,
                "reply_to": message.reply_to_msg_id, "forwarded": bool(message.fwd_from),
                "via_bot": bool(message.via_bot_id), "out": message.out, "text": message.raw_text or ""}
            if valid_ack(candidate, operation, self.config["hex_bot_id"], self.config["destination_chat_id"]):
                return message.id
        return None

    @guarded
    async def reconcile(self, ledger):
        """Observe one existing operation; this method cannot create or send one."""
        operation = ledger.pending()
        if not operation:
            return
        if operation["error"] == "receiver_failed" and operation["reply_id"]:
            return  # A recorded rejection is final for this attempt.
        if not operation["message_id"]:
            found = await self.recover(operation)
            if not found:
                ledger.uncertain(operation["id"])
                return
            ledger.sent(operation["id"], found)
            operation = ledger.pending()
        reply = await self.acknowledgement(operation)
        if reply:
            ledger.acknowledge(operation["id"], reply)
            return
        async for message in self.client.iter_messages(self.destination, limit=200):
            if message.id <= operation["message_id"]:
                break
            if (message.chat_id == self.config["destination_chat_id"]
                    and message.sender_id == self.config["hex_bot_id"]
                    and message.reply_to_msg_id == operation["message_id"]
                    and not message.fwd_from and not message.via_bot_id and not message.out
                    and message.raw_text == "⚠️ 操作失敗，請稍後再試。"):
                with ledger.conn:
                    ledger.conn.execute("UPDATE operations SET state='uncertain',error='receiver_failed',reply_id=? WHERE id=?", (message.id, operation["id"]))
                return
        if int(time.time()) - operation["created_at"] > 120 and operation["error"] != "receiver_failed":
            ledger.uncertain(operation["id"])

    @guarded
    async def control_peers(self, users, group_ok):
        me = await self.client.get_me()
        if not me or me.bot or me.id != self.config["account_id"]:
            raise StopBridge("Wrong Telegram account")
        # Recover only pending replies and authorized conversations after a restart.
        # Ordinary private messages arrive through events, never a history scan.
        if int(time.time()) - self.peers_checked_at >= 300:
            wanted = set(users)
            if self.ledger is not None:
                wanted.update(r[0] for r in self.ledger.conn.execute(
                    "SELECT DISTINCT chat_id FROM watch_outbox WHERE state IN ('queued','sending') AND chat_id>0"))
            async for dialog in self.client.iter_dialogs(limit=100):
                if dialog.id in wanted and dialog.is_user and not getattr(dialog.entity, "bot", False):
                    self.remember_peer(dialog.id, dialog.entity)
            self.peers_checked_at = int(time.time())
        peers = dict(self.peers)
        if group_ok:
            peers[self.config["destination_chat_id"]] = self.destination
            # Resolve the owner's access hash from verified membership for a first DM.
            if self.config["operator_id"] not in peers:
                members = await self.client.get_participants(self.destination, limit=10)
                for member in members:
                    if member.id == self.config["operator_id"] and not member.bot:
                        peers[member.id] = member
        return peers

    @guarded
    async def control_messages(self, peer, cursor):
        options = {"limit": 1} if cursor is None else {"limit": 50, "min_id": cursor, "reverse": True}
        return [{"id": m.id, "actor": m.sender_id, "text": m.raw_text or "",
                 "date": int(m.date.timestamp()),
                 "trusted": not (m.fwd_from or m.via_bot_id or m.out or m.edit_date or m.action)}
                async for m in self.client.iter_messages(peer, **options)]

    def install_control_handler(self, handler):
        from telethon import events

        async def incoming(event):
            message = event.message
            if (message.chat_id > 0 and self.config.get("owner_only_private")
                    and message.sender_id != self.config["operator_id"]):
                return
            if message.chat_id != message.sender_id and message.chat_id != self.config["destination_chat_id"]:
                return
            if message.fwd_from or message.via_bot_id or message.out or message.edit_date or message.action:
                return
            # Events normally include both entities. Do not fetch missing senders
            # while Telegram is rate limiting us.
            sender = event.sender
            peer = event.input_chat
            if not sender or not peer or getattr(sender, "bot", False):
                return
            await handler(message.chat_id, message.id, message.sender_id,
                          message.raw_text or "", peer, int(message.date.timestamp()))

        self.client.add_event_handler(incoming, events.NewMessage(incoming=True))

    @guarded
    async def send_notice(self, peer, notice):
        from telethon import errors, functions
        try:
            result = await asyncio.wait_for(self.client(functions.messages.SendMessageRequest(
                peer=await self.client.get_input_entity(peer), message=notice["body"],
                random_id=notice["random_id"], no_webpage=True)), timeout=25)
        except errors.FloodWaitError as error:
            raise Flood(error.seconds) from None
        if getattr(result, "id", None):
            return result.id
        for update in getattr(result, "updates", []):
            if getattr(update, "random_id", None) == notice["random_id"]:
                return update.id
        raise StopBridge("Notice accepted without a message mapping")

    @guarded
    async def recover_notice(self, peer, notice):
        matches = [m.id async for m in self.client.iter_messages(peer, limit=100)
                   if m.out and m.sender_id == self.config["account_id"] and not m.fwd_from
                   and m.raw_text == notice["body"] and int(m.date.timestamp()) >= notice["created_at"] - 5]
        if len(matches) > 1:
            raise StopBridge("Duplicate notices; inspect the conversation")
        return matches[0] if matches else None


async def cycle(config, ledger, binding, transport, snapshot_reader):
    snapshot = snapshot_reader()
    ledger.check(snapshot, binding)
    await transport.check_destination()
    pending = ledger.pending()
    if pending:
        if not pending["message_id"]:
            found = await transport.recover(pending)
            if found:
                ledger.sent(pending["id"], found)
                pending = ledger.pending()
            else:
                ledger.uncertain(pending["id"])
                raise StopBridge("A previous send is unconfirmed; no command was resent")
        reply = await transport.acknowledgement(pending)
        if reply:
            ledger.acknowledge(pending["id"], reply)
            return "confirmed"
        if int(time.time()) - pending["created_at"] > 120:
            ledger.uncertain(pending["id"])
            raise StopBridge("hexDrill has not confirmed the command; inspect its reply")
        return "awaiting_reply"
    changes = ledger.plan(snapshot, binding)
    if not changes:
        return "caught_up"
    # Re-read immediately before claiming a send. A database error is never an empty blacklist.
    latest = snapshot_reader()
    latest_changes = ledger.plan(latest, binding)
    if not latest_changes:
        return "caught_up"
    operation = ledger.create(latest_changes[0])
    if operation is None:
        return "cooldown"
    try:
        message = await asyncio.wait_for(transport.send(operation), timeout=25)
        ledger.sent(operation["id"], message)
    except Flood as error:
        ledger.flood(operation["id"], error.seconds)
        return "flood_wait"
    except BaseException:
        ledger.uncertain(operation["id"])
        raise
    return "sent_waiting_for_reply"
