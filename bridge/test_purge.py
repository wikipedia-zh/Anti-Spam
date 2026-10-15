import asyncio
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

from control import Control, Watch
from core import StopBridge
from purge import run, interrupted
from telegram_client import Telegram, Flood
from test_bridge import Fixture
from test_control import FakeTransport


class Purge(Fixture):
    def setUp(self):
        super().setUp()
        self.config = {'source_db': str(self.source), 'operator_id': 500, 'account_id': 11,
                       'hex_bot_id': 12, 'destination_chat_id': -10001, 'owner_only_private': True}
        self.c = Control(self.config, self.ledger)
        with self.c.db:
            self.c.advance(500, 0)
            self.c.advance(-10001, 0)

    def test_only_authorized_users_can_queue_and_replay_cannot_repeat_deletion(self):
        self.c.handle(500, 1, 500, '/sudo 600')
        self.c.handle(-10001, 2, 700, '/purge')
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM purge_jobs').fetchone()[0], 0)
        self.c.handle(-10001, 3, 500, '/purge', trusted=False)
        self.c.handle(-10001, 4, 500, '/purge')
        transport = PurgeTransport()
        asyncio.run(run(self.c, transport, {-10001: -10001}))
        self.c.handle(-10001, 4, 500, '/purge')
        asyncio.run(run(self.c, transport, {-10001: -10001}))
        self.assertEqual(transport.deleted, [10, 12, 14])
        self.assertEqual(self.c.db.execute('SELECT state,deleted,end_id FROM purge_jobs').fetchone()[:], ('done', 3, 14))
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM operations').fetchone()[0], 0)

    def test_partial_failure_stops_without_retrying_or_expanding_range(self):
        self.c.handle(-10001, 4, 500, '/purge')
        transport = PurgeTransport(fail=True)
        asyncio.run(run(self.c, transport, {-10001: -10001}))
        asyncio.run(run(self.c, transport, {-10001: -10001}))
        self.assertEqual(transport.deleted, [10, 12])
        self.assertEqual(self.c.db.execute('SELECT state,deleted FROM purge_jobs').fetchone()[:], ('stopped', 2))
        self.assertEqual(transport.attempts, 2)

    def test_restart_does_not_resume_an_old_delete_request(self):
        self.c.handle(-10001, 4, 500, '/purge')
        interrupted(self.c)
        transport = PurgeTransport()
        asyncio.run(run(self.c, transport, {-10001: -10001}))
        self.assertEqual(transport.deleted, [])
        self.assertEqual(self.c.db.execute('SELECT state FROM purge_jobs').fetchone()[0], 'stopped')

    def test_event_executes_once_and_sends_completion(self):
        transport = PurgeTransport()
        watch = Watch(self.config, self.ledger, transport)
        asyncio.run(watch.incoming(-10001, 4, 500, '/purge', -10001))
        self.assertEqual(transport.deleted, [10, 12, 14])
        self.assertEqual(transport.sent, [-10001])

    def test_bounds_require_fresh_owner_reply_in_same_chat(self):
        command = SimpleNamespace(chat_id=500, sender_id=500, out=False, fwd_from=None,
            via_bot_id=None, edit_date=None, raw_text='/purge', date=datetime.now(timezone.utc),
            reply_to_msg_id=10, reply_to=SimpleNamespace(reply_to_peer_id=None))
        target = SimpleNamespace(chat_id=500, id=10)
        class Client:
            async def get_messages(self, peer, ids=None, limit=None):
                return command if ids == 14 else target if ids == 10 else [SimpleNamespace(chat_id=500, id=15)]
        transport = Telegram(Client(), self.config)
        fake_telethon = SimpleNamespace(utils=SimpleNamespace(get_peer_id=lambda peer: peer))
        with patch.dict('sys.modules', {'telethon': fake_telethon}):
            self.assertEqual(asyncio.run(transport.purge_bounds(500, 500, 14, 500)), (10, 15))
            for attr, value in [('sender_id', 600), ('edit_date', datetime.now(timezone.utc)),
                                ('reply_to_msg_id', None), ('date', datetime.now(timezone.utc)-timedelta(minutes=3))]:
                old = getattr(command, attr)
                setattr(command, attr, value)
                with self.assertRaises(StopBridge):
                    asyncio.run(transport.purge_bounds(500, 500, 14, 500))
                setattr(command, attr, old)
            target.chat_id = 600
            with self.assertRaises(StopBridge):
                asyncio.run(transport.purge_bounds(500, 500, 14, 500))

    def test_private_id_gaps_never_delete_other_conversations(self):
        class Client:
            wrong_chat = False
            async def iter_messages(self, peer, **kwargs):
                self.options = kwargs
                yield SimpleNamespace(chat_id=500, id=10)
                yield SimpleNamespace(chat_id=600 if self.wrong_chat else 500, id=14)
            delete_messages = AsyncMock(return_value=[])
        client = Client()
        transport = Telegram(client, self.config)
        ids = asyncio.run(transport.purge_batch(500, 500, 9, 14))
        asyncio.run(transport.delete_purge_batch(500, 500, ids, 500))
        client.delete_messages.assert_awaited_once_with(500, [10, 14], revoke=True)
        self.assertEqual(client.options['max_id'], 15)
        client.wrong_chat = True
        with self.assertRaises(StopBridge):
            asyncio.run(transport.purge_batch(500, 500, 9, 14))

    def test_delegate_purge_works_in_other_groups_and_private_chats(self):
        self.c.handle(500, 1, 500, '/sudo 600')
        transport = PurgeTransport()
        watch = Watch(self.config, self.ledger, transport)
        async def exercise():
            await watch.incoming(-20002, 4, 600, '/purge', -20002)
            await watch.incoming(600, 5, 600, '/purge', 600)
            await watch.incoming(600, 6, 600, '/sync 100', 600)
            await watch.incoming(-30003, 7, 700, '/purge', -30003)
        asyncio.run(exercise())
        self.assertEqual(transport.sent, [-20002, 600])
        self.assertEqual(transport.deleted, [10, 12, 14, 10, 12, 14])
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM purge_jobs').fetchone()[0], 2)

    def test_revocation_stops_remaining_batches(self):
        self.c.handle(500, 1, 500, '/sudo 600')
        self.c.handle(-10001, 2, 600, '/purge')
        control = self.c
        class Revoked(PurgeTransport):
            async def delete_purge_batch(self, peer, chat, ids, actor):
                await super().delete_purge_batch(peer, chat, ids, actor)
                with control.db:
                    control.db.execute('DELETE FROM sudo_users WHERE user_id=600')
        transport = Revoked()
        asyncio.run(run(self.c, transport, {-10001:-10001}))
        self.assertEqual(transport.deleted, [10, 12])
        self.assertEqual(self.c.db.execute('SELECT state FROM purge_jobs').fetchone()[0], 'stopped')

    def test_event_routing_allows_only_authorized_purge_outside_control_chat(self):
        self.c.handle(500, 1, 500, '/sudo 600')
        class Client:
            def add_event_handler(self, handler, event):
                self.handler = handler
        client = Client()
        transport = Telegram(client, self.config)
        transport.bind_ledger(self.ledger)
        handler = AsyncMock()
        with patch.dict('sys.modules', {'telethon': SimpleNamespace(events=SimpleNamespace(NewMessage=lambda **kw: kw))}):
            transport.install_control_handler(handler)
        for chat, actor, text in [(-20002,600,'/purge'), (600,600,'/purge'),
                                  (-30003,700,'/purge'), (-20002,600,'/sync 100'), (600,600,'hello')]:
            event = SimpleNamespace(sender=SimpleNamespace(bot=False), input_chat=chat,
                message=SimpleNamespace(chat_id=chat, sender_id=actor, id=10, raw_text=text,
                    date=datetime.now(timezone.utc), fwd_from=None, via_bot_id=None,
                    out=False, edit_date=None, action=None))
            asyncio.run(client.handler(event))
        self.assertEqual(handler.await_count, 2)


class PurgeTransport(FakeTransport):
    def __init__(self, fail=False):
        super().__init__(dm_bad=False)
        self.deleted, self.attempts, self.fail = [], 0, fail
    async def purge_bounds(self, peer, chat, command, actor):
        return 10, 14
    async def purge_batch(self, peer, chat, cursor, end):
        return [i for i in [10, 12, 14, 15, 16] if cursor < i <= end][:2]
    async def delete_purge_batch(self, peer, chat, ids, actor):
        self.attempts += 1
        if self.fail and self.attempts == 2:
            raise Flood(300)
        self.deleted.extend(ids)
