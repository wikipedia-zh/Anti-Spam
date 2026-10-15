import asyncio
from datetime import datetime, timezone
import time
from types import SimpleNamespace
import unittest

from control import Control, Watch, lookup
from core import StopBridge
from telegram_client import Telegram
from test_bridge import Fixture


class Access(Fixture):
    def setUp(self):
        super().setUp()
        self.config = {"source_db": str(self.source), "operator_id": 500, "account_id": 11,
                       "hex_bot_id": 12, "destination_chat_id": -10001}
        self.c = Control(self.config, self.ledger)
        with self.c.db:
            self.c.advance(500, 0)
            self.c.advance(-10001, 0)

    def test_only_owner_grants_and_revokes_persistently(self):
        self.c.handle(-10001, 1, 600, "/sudo 700")
        self.assertFalse(self.c.authorized(700))
        self.c.handle(500, 1, 500, "/sudo 600")
        self.assertTrue(Control(self.config, self.ledger).authorized(600))
        self.c.handle(-10001, 2, 600, "/sudo 700")
        self.assertFalse(self.c.authorized(700))
        self.c.handle(-10001, 3, 600, "/spb_sync 100")
        self.c.handle(500, 2, 500, "/sudo del 600")
        self.assertFalse(self.c.authorized(600))
        self.assertEqual(self.c.db.execute("SELECT state FROM watch_outbox WHERE key='reply:-10001:3'").fetchone()[0], "cancelled")
        # Replaying the original grant after revocation cannot grant again.
        self.c.handle(500, 1, 500, "/sudo 600")
        self.assertFalse(self.c.authorized(600))
        with self.assertRaises(StopBridge):
            Control(dict(self.config, operator_id=600), self.ledger)

    def test_untrusted_messages_and_other_groups_are_silent(self):
        self.c.handle(-10001, 1, 500, "/sudo 600", trusted=False)
        with self.c.db:
            self.c.advance(-10002, 0)
        self.c.handle(-10002, 1, 500, "/sudo 600")
        self.assertFalse(self.c.authorized(600))
        self.assertEqual(self.c.db.execute("SELECT count(*) FROM watch_outbox").fetchone()[0], 0)

    def operation(self):
        case = self.add()
        op = self.ledger.create(("ban", 100, case))
        self.ledger.sent(op["id"], 10)
        self.ledger.uncertain(op["id"])
        return op

    def test_lookup_reports_actual_status_and_rejects_collisions(self):
        op = self.operation()
        self.c.refresh()
        body = lookup(self.c.db, "sync:" + op["id"].replace("-", "")[:12])[0][0]
        self.assertIn("未確認成功", body)
        self.assertIn("1.0000", body)
        self.assertIn("auto_banned", body)
        self.assertEqual(len(lookup(self.c.db, "#" + op["case_id"][:8])), 1)
        self.assertEqual(len(lookup(self.c.db, "100")), 1)
        with self.c.db:
            self.c.db.execute("INSERT INTO operation_reports SELECT substr(id,1,length(id)-1)||?,user_id,case_id,body,created_at,updated_at FROM operation_reports", ("0" if op["id"][-1] != "0" else "1",))
        with self.assertRaises(ValueError):
            lookup(self.c.db, "sync:" + op["id"].replace("-", "")[:12])
        with self.assertRaises(ValueError):
            lookup(self.c.db, "' OR 1=1")

    def test_failure_alerts_are_durable_and_cancel_if_success_arrives(self):
        op = self.operation()
        self.c.refresh()
        Control(self.config, self.ledger).refresh()
        self.assertEqual(self.c.db.execute("SELECT count(*) FROM watch_outbox").fetchone()[0], 2)
        self.ledger.acknowledge(op["id"], 11)
        self.c.refresh()
        self.assertEqual(self.c.db.execute("SELECT count(*) FROM watch_outbox WHERE state='queued'").fetchone()[0], 0)

    def test_broken_source_does_not_change_confirmed_ownership(self):
        self.operation()
        self.owned(100)
        self.db.execute("DROP TABLE cases")
        self.db.commit()
        with self.assertRaises(Exception):
            self.c.refresh()
        self.assertEqual(self.c.db.execute("SELECT active FROM owned WHERE user_id=100").fetchone()[0], 1)
        self.assertEqual(self.c.db.execute("SELECT count(*) FROM operation_reports").fetchone()[0], 0)

    def test_watch_does_not_send_bans_and_delivers_group_if_dm_fails(self):
        self.operation()
        transport = FakeTransport()
        watch = Watch(dict(self.config, live=True), self.ledger, transport)
        asyncio.run(watch.tick())
        self.assertEqual(transport.sent, [-10001])
        asyncio.run(watch.tick())
        self.assertEqual(transport.sent, [-10001])
        self.assertEqual(self.c.db.execute("SELECT count(*) FROM operations").fetchone()[0], 1)
        self.assertEqual(self.c.db.execute("SELECT state FROM watch_outbox WHERE chat_id=500").fetchone()[0], "sending")

    def test_group_compromise_blocks_group_but_warns_owner(self):
        self.operation()
        transport = FakeTransport(group_bad=True, dm_bad=False)
        asyncio.run(Watch(self.config, self.ledger, transport).tick())
        self.assertTrue(transport.sent)
        self.assertTrue(all(chat == 500 for chat in transport.sent))

    def test_receiver_failure_requires_pinned_reply(self):
        op = self.operation()
        operation = self.ledger.pending()
        command = SimpleNamespace(sender_id=11, out=True, raw_text=operation["command"])
        def candidate(**kwargs):
            values = dict(id=11, chat_id=-10001, sender_id=12, reply_to_msg_id=10,
                          fwd_from=None, via_bot_id=None, out=False, raw_text="⚠️ 操作失敗，請稍後再試。")
            values.update(kwargs)
            return SimpleNamespace(**values)
        class Client:
            async def get_messages(self, *args, **kwargs):
                return command
            async def iter_messages(self, *args, **kwargs):
                for item in self.items:
                    yield item
        client = Client()
        telegram = Telegram(client, self.config)
        for forged in (candidate(sender_id=600), candidate(reply_to_msg_id=9), candidate(fwd_from=True), candidate(via_bot_id=1), candidate(out=True), candidate(chat_id=-20000)):
            client.items = [forged]
            asyncio.run(telegram.reconcile(self.ledger))
            self.assertNotEqual(self.ledger.pending()["error"], "receiver_failed")
        client.items = [candidate()]
        asyncio.run(telegram.reconcile(self.ledger))
        self.assertEqual(self.ledger.pending()["error"], "receiver_failed")
        self.assertEqual(self.ledger.pending()["reply_id"], 11)
        self.assertEqual(self.c.db.execute("SELECT count(*) FROM owned").fetchone()[0], 0)


class FakeTransport:
    def __init__(self, group_bad=False, dm_bad=True):
        self.sent, self.group_bad, self.dm_bad = [], group_bad, dm_bad
    async def check_destination(self):
        if self.group_bad:
            raise StopBridge("changed members")
    async def reconcile(self, ledger):
        pass
    async def control_peers(self, users, group_ok):
        return {user: user for user in users} | ({-10001: -10001} if group_ok else {})
    async def control_messages(self, peer, cursor):
        return []
    async def send_notice(self, peer, notice):
        if peer == 500 and self.dm_bad:
            raise TimeoutError()
        self.sent.append(peer)
        return len(self.sent)
    async def recover_notice(self, peer, notice):
        return None
    async def send(self, *args):
        raise AssertionError("watch must never send moderation commands")


if __name__ == "__main__":
    unittest.main()
