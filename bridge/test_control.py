import asyncio
from datetime import datetime, timezone
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch, AsyncMock

from control import Control, Watch, lookup
from core import StopBridge
from telegram_client import Telegram, Flood
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

    def test_live_commands_reply_without_waiting_for_poll_and_do_not_replay(self):
        transport = FakeTransport(dm_bad=False)
        watch = Watch(self.config, self.ledger, transport)
        async def exercise():
            await watch.incoming(500, 1, 500, "/sudo list", 500)
            await watch.incoming(500, 1, 500, "/sudo list", 500)
            await watch.incoming(600, 2, 600, "/sudo 700", 600)
        asyncio.run(exercise())
        self.assertEqual(transport.sent, [500])
        self.assertFalse(self.c.authorized(700))

    def test_live_group_commands_require_fresh_membership(self):
        watch = Watch(self.config, self.ledger, FakeTransport(group_bad=True))
        asyncio.run(watch.incoming(-10001, 1, 500, "/sudo 600", -10001))
        self.assertFalse(self.c.authorized(600))

    def test_old_private_dialogs_are_not_polled(self):
        class ManyDialogs(FakeTransport):
            def __init__(self):
                super().__init__(dm_bad=False)
                self.polled = []
            async def control_peers(self, users, group_ok):
                return await super().control_peers(users, group_ok) | {i: i for i in range(1000, 2000)}
            async def control_messages(self, peer, cursor):
                self.polled.append(peer)
                return []
        transport = ManyDialogs()
        watch = Watch(dict(self.config, dm_enabled=True), self.ledger, transport)
        asyncio.run(watch.tick())
        asyncio.run(watch.tick())
        self.assertEqual(set(transport.polled), {500, -10001})
        self.assertEqual(len(transport.polled), 2)

    def test_flood_stops_cycle_without_membership_warning(self):
        class Limited(FakeTransport):
            async def check_destination(self):
                raise Flood(600)
            async def control_peers(self, *args):
                raise AssertionError("must stop after first rate limit")
        asyncio.run(Watch(self.config, self.ledger, Limited()).tick())
        self.assertEqual(self.c.db.execute("SELECT count(*) FROM watch_outbox").fetchone()[0], 0)

    def test_new_private_event_keeps_peer_for_delayed_model_reply(self):
        transport = FakeTransport(dm_bad=False)
        watch = Watch(dict(self.config, dm_enabled=True), self.ledger, transport)
        asyncio.run(watch.incoming(600, 1, 600, "hello", 600))
        with self.c.db:
            self.c.db.execute("UPDATE watch_inbox SET status='ham'")
        asyncio.run(watch.tick())
        self.assertEqual(transport.sent, [600])
        self.assertFalse(self.c.authorized(600))

    def test_recovery_flood_never_requeues_an_uncertain_send(self):
        class Limited(FakeTransport):
            async def recover_notice(self, *args):
                raise Flood(600)
        with self.c.db:
            self.c.enqueue("already-sending", 500, "test")
            self.c.db.execute("UPDATE watch_outbox SET state='sending'")
        watch = Watch(self.config, self.ledger, Limited())
        with self.assertRaises(Flood):
            asyncio.run(watch.deliver({500: 500}))
        row = self.c.db.execute("SELECT state,next_at FROM watch_outbox").fetchone()
        self.assertEqual(row[0], "sending")
        self.assertGreater(row[1], int(time.time()) + 590)

    def test_transport_persists_flood_and_blocks_other_rpc_after_restart(self):
        class FloodWaitError(Exception):
            seconds = 600
        class Client:
            calls = 0
            async def iter_messages(self, *args, **kwargs):
                self.calls += 1
                raise FloodWaitError()
                yield
            async def get_me(self):
                raise AssertionError("cooldown must cover all RPC methods")
        client = Client()
        transport = Telegram(client, self.config)
        transport.bind_ledger(self.ledger)
        with patch.dict("sys.modules", {"telethon.errors": SimpleNamespace(FloodWaitError=FloodWaitError)}):
            with self.assertRaises(Flood):
                asyncio.run(transport.control_messages(500, 0))
        restarted = Telegram(client, self.config)
        restarted.bind_ledger(self.ledger)
        with self.assertRaises(Flood):
            asyncio.run(restarted.control_peers({500}, True))
        self.assertEqual(client.calls, 1)
        self.assertGreater(self.ledger.get("telegram_wait_until"), int(time.time()) + 590)

    def test_peer_discovery_is_bounded_and_does_not_collect_old_dms(self):
        class Client:
            scans = 0
            async def get_me(self):
                return SimpleNamespace(id=11, bot=False)
            async def iter_dialogs(self, *, limit):
                self.scans += 1
                self.limit = limit
                for user in range(500, 500 + limit):
                    yield SimpleNamespace(id=user, is_user=True, entity=SimpleNamespace(id=user, bot=False))
        client = Client()
        transport = Telegram(client, dict(self.config, dm_enabled=True))
        self.assertEqual(set(asyncio.run(transport.control_peers({500}, False))), {500})
        self.assertEqual(set(asyncio.run(transport.control_peers({500}, False))), {500})
        self.assertEqual(client.scans, 1)
        self.assertEqual(client.limit, 100)

    def test_owner_only_private_keeps_owner_and_group_commands(self):
        config = dict(self.config, owner_only_private=True, dm_enabled=False)
        transport = FakeTransport(dm_bad=False)
        watch = Watch(config, self.ledger, transport)
        async def exercise():
            await watch.incoming(500, 1, 500, "/sudo 600", 500)
            await watch.incoming(600, 2, 600, "/sync 100", 600)
            await watch.incoming(700, 3, 700, "casino", 700)
            await watch.incoming(-10001, 4, 600, "/spb_sync 100", -10001)
            await watch.incoming(500, 5, 500, "/sudo list", 500)
        asyncio.run(exercise())
        self.assertEqual(transport.sent, [500, -10001, 500])
        self.assertTrue(watch.control.authorized(600))
        self.assertEqual(watch.control.db.execute("SELECT count(*) FROM watch_inbox").fetchone()[0], 0)
        self.assertNotIn(600, watch.incoming_peers)
        self.assertNotIn(700, watch.incoming_peers)
        body = watch.control.db.execute("SELECT body FROM watch_outbox WHERE key='reply:500:1'").fetchone()[0]
        self.assertIn("指定群組", body)

    def test_owner_only_private_cancels_old_replies_without_reading_history(self):
        class NoHistory(FakeTransport):
            async def control_messages(self, *args):
                raise AssertionError("non-owner private history must not be read")
        watch = Watch(dict(self.config, owner_only_private=True), self.ledger, NoHistory(dm_bad=False))
        with self.c.db:
            self.c.enqueue("old-greeting", 600, "hello")
            self.c.enqueue("owner-alert", 500, "warning")
        asyncio.run(watch.poll(600, 600))
        asyncio.run(watch.deliver({500:500, 600:600}))
        self.assertEqual(watch.transport.sent, [500])
        self.assertEqual(self.c.db.execute("SELECT state FROM watch_outbox WHERE key='old-greeting'").fetchone()[0], "cancelled")

    def test_owner_short_update_replies_without_profile_or_history_fetch(self):
        class Client:
            def add_event_handler(self, handler, event):
                self.handler = handler
        client = Client()
        transport = Telegram(client, dict(self.config, owner_only_private=True, dm_enabled=False))
        transport.send_notice = AsyncMock(return_value=90)
        watch = Watch(transport.config, self.ledger, transport)
        transport.remember_peer(500, "cached-owner-peer")
        event = SimpleNamespace(sender=None, input_chat=None,
            get_sender=AsyncMock(side_effect=AssertionError("owner profile is unnecessary")),
            get_input_chat=AsyncMock(side_effect=AssertionError("use cached peer")),
            message=SimpleNamespace(chat_id=500, sender_id=500, id=1, raw_text="/sudo list",
                date=datetime.now(timezone.utc), fwd_from=None, via_bot_id=None,
                out=False, edit_date=None, action=None))
        events = SimpleNamespace(NewMessage=lambda **kwargs: kwargs)
        with patch.dict("sys.modules", {"telethon": SimpleNamespace(events=events)}):
            transport.install_control_handler(watch.incoming)
        asyncio.run(client.handler(event))
        transport.send_notice.assert_awaited_once()
        self.assertEqual(transport.send_notice.call_args.args[0], "cached-owner-peer")
        self.assertEqual(self.c.db.execute("SELECT state FROM watch_outbox WHERE key='reply:500:1'").fetchone()[0], "sent")
        self.assertEqual(self.ledger.get("watch_last_owner_event")["message_id"], 1)
        event.message.chat_id = event.message.sender_id = 600
        asyncio.run(client.handler(event))
        transport.send_notice.assert_awaited_once()

    def test_owner_short_update_resolves_only_missing_peer_and_honors_cooldown(self):
        transport = Telegram(None, self.config)
        transport.bind_ledger(self.ledger)
        event = SimpleNamespace(sender=None, input_chat=None,
            message=SimpleNamespace(sender_id=500, chat_id=500),
            get_sender=AsyncMock(side_effect=AssertionError("no owner profile lookup")),
            get_input_chat=AsyncMock(return_value="resolved-owner-peer"))
        self.assertEqual(asyncio.run(transport.event_peer(event)), "resolved-owner-peer")
        event.get_input_chat.assert_awaited_once()
        transport.wait_until = int(time.time()) + 300
        with self.assertRaises(Flood):
            asyncio.run(transport.event_peer(event))
        event.get_input_chat.assert_awaited_once()


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
