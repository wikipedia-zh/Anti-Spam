import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from test_bridge import Fixture
from bridge import binding
from control import Control, Watch
from test_control import FakeTransport
from core import StopBridge
from telegram_client import Telegram, Flood
from delivery_queue import state, tick
from burst_queue import request


class Burst(Fixture):
    def setUp(self):
        super().setUp()
        self.config = {'source_db':str(self.source),'operator_id':500,'account_id':11,
                       'hex_bot_id':12,'destination_chat_id':-10001,'allowed_observers':[500],
                       'receiver_scope_confirmed':True,'owner_only_private':True}
        self.one = self.add(100)
        self.two = self.add(101)
        self.legacy = self.add(102)
        self.ledger.initialize(self.snap(),binding(self.config))
        with self.ledger.conn:
            self.ledger.conn.execute('UPDATE seen SET approved=1 WHERE case_id IN (?,?)',(self.one,self.two))
        self.c = Control(self.config,self.ledger)
        self.t = SimpleNamespace(reconcile_batch=AsyncMock(), confirmed_failure=AsyncMock(return_value=True),
                                 send=AsyncMock(side_effect=[10,20,30]))

    def start(self):
        with self.c.db:
            return request(self.c,500,'test-batch')

    def run_tick(self):
        asyncio.run(tick(self.c,self.t))

    def test_sends_batch_without_ack_and_never_replays_sent_after_restart(self):
        self.start()
        self.run_tick()
        self.assertEqual(self.t.send.await_count,2)
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM owned').fetchone()[0],0)
        self.c = Control(self.config,self.ledger)
        self.run_tick()
        self.assertEqual(self.t.send.await_count,2)
        for op in self.c.db.execute('SELECT id,message_id FROM operations').fetchall():
            self.ledger.acknowledge(op['id'],op['message_id']+1)
        self.run_tick()
        self.assertEqual(state(self.c)['status'],'done')
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM owned WHERE active=1').fetchone()[0],2)
        self.assertEqual(self.c.db.execute('SELECT approved FROM seen WHERE case_id=?',(self.legacy,)).fetchone()[0],0)

    def test_unknown_send_stops_without_sending_next_or_retrying(self):
        self.start();self.t.send.side_effect=TimeoutError
        with self.assertRaises(TimeoutError):self.run_tick()
        self.run_tick()
        self.assertEqual(self.t.send.await_count,1)
        self.assertEqual(state(self.c)['status'],'paused')
        self.assertIn('不明',self.start())

    def test_flood_wait_keeps_batch_and_blocks_further_sends(self):
        self.start();self.t.send.side_effect=Flood(30)
        with self.assertRaises(Flood):self.run_tick()
        self.run_tick()
        self.assertEqual(self.t.send.await_count,1)
        self.assertEqual(state(self.c)['status'],'running')

    def test_observed_failure_stops_remaining_items(self):
        with self.c.db:self.c.db.execute('UPDATE seen SET approved=1')
        self.start()
        async def send(op):
            if op['user_id']==101:
                with self.c.db:self.c.db.execute("UPDATE operations SET state='uncertain',error='receiver_failed',reply_id=11 WHERE user_id=100")
            return op['user_id']
        self.t.send.side_effect=send
        self.run_tick()
        self.assertEqual(self.t.send.await_count,2)
        self.assertEqual(state(self.c)['status'],'paused')

    def test_revoked_case_not_sent_and_new_approvals_do_not_expand_batch(self):
        self.start();self.retire(self.one)
        with self.c.db:self.c.db.execute('UPDATE seen SET approved=1')
        self.run_tick()
        self.assertEqual(self.t.send.await_count,1)
        self.assertEqual(self.t.send.call_args.args[0]['user_id'],101)

    def test_live_receipt_requires_pinned_reply_and_updates_ownership(self):
        self.start();self.run_tick()
        t=Telegram(None,self.config);t.bind_ledger(self.ledger)
        reply=SimpleNamespace(id=11,chat_id=-10001,sender_id=999,reply_to_msg_id=10,
                              fwd_from=None,via_bot_id=None,out=False,
                              raw_text='ok\n📋 阻擋清單：\n✅ +spb：已加入，封鎖（永久）')
        t.observe_reply(reply)
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM owned').fetchone()[0],0)
        reply.sender_id=12;t.observe_reply(reply);t.observe_reply(reply)
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM owned WHERE active=1').fetchone()[0],1)

    def test_destination_flood_is_rechecked_after_cooldown_not_reported_as_compromise(self):
        t=FakeTransport(dm_bad=False)
        t.check_destination=AsyncMock(side_effect=[Flood(1),None])
        w=Watch(self.config,self.ledger,t)
        asyncio.run(w.tick())
        self.assertEqual(w.group_checked_at,0)
        asyncio.run(w.tick())
        self.assertTrue(w.group_ok)
        self.assertEqual(t.check_destination.await_count,2)
        self.assertEqual(self.c.db.execute("SELECT count(*) FROM watch_outbox WHERE key='destination-unavailable'").fetchone()[0],0)

    def test_reversal_after_send_removes_confirmed_batch_entry(self):
        self.start();self.run_tick()
        first=self.c.db.execute('SELECT id FROM operations WHERE user_id=100').fetchone()[0]
        self.retire(self.one)
        self.ledger.acknowledge(first,11)
        self.run_tick()
        self.assertEqual(self.t.send.await_count,3)
        self.assertEqual(self.t.send.call_args.args[0]['action'],'unban')
        self.assertEqual(self.t.send.call_args.args[0]['user_id'],100)
