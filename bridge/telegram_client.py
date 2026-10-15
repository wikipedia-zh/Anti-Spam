"""The only module allowed to send Telegram messages."""
import asyncio
import json
import os
from pathlib import Path
import time

from core import HEX_USERNAME, StopBridge, valid_ack


class Flood(Exception):
    def __init__(self, seconds):
        self.seconds = seconds


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
        receive_updates=False, device_model="SPB bridge", app_version="1.0")
    client.session.save_entities = False
    return client


class Telegram:
    def __init__(self, client, config):
        self.client, self.config = client, config
        self.destination = None

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
