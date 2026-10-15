import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

from control import Control, Watch
from core import StopBridge
from spy import capture, command, excluded, joins, target, interrupted
from spy_delivery import claim, candidates, tick
from test_bridge import Fixture


class Spy(Fixture):
    def setUp(self):
        super().setUp()
        self.config = {'source_db':str(self.source),'operator_id':500,'account_id':11,'hex_bot_id':12,
                       'destination_chat_id':-10001,'spy_enabled':True,'spb_bot_id':13,
                       'spy_excluded_chats':[-10002],'owner_only_private':True,'receiver_scope_confirmed':True}
        self.c = Control(self.config,self.ledger)
        with self.c.db:
            self.c.advance(500,0)
            self.c.advance(600,0)
            self.c.advance(-10003,0)
            self.c.db.execute('INSERT INTO spy_groups VALUES (-10003,500,0)')

    def test_membership_without_join_is_never_scanned(self):
        capture(self.c,-10004,1,100,'spam',int(time.time()))
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM watch_spy').fetchone()[0],0)
        self.c.handle(500,1,500,'/ignore del -10004')
        self.assertTrue(excluded(self.c,-10004))

    def test_capture_ignores_private_reserved_protected_and_duplicates(self):
        now = int(time.time())
        for chat,actor in [(100,100),(-10001,100),(-10002,100),(-10003,500),(-10003,777000)]:
            capture(self.c,chat,1,actor,'spam',now)
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM watch_spy').fetchone()[0],0)
        capture(self.c,-10003,1,100,'spam',now)
        capture(self.c,-10003,1,100,'edited',now)
        self.assertEqual(self.c.db.execute('SELECT text FROM watch_spy').fetchone()[0],'spam')
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM operations').fetchone()[0],0)
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM watch_outbox').fetchone()[0],0)

    def test_ignore_clears_pending_and_fixed_exclusions_cannot_be_removed(self):
        capture(self.c,-10003,1,100,'spam',int(time.time()))
        self.c.handle(500,1,500,'/ignore -10003')
        self.assertTrue(excluded(self.c,-10003))
        self.assertEqual(tuple(self.c.db.execute('SELECT text,status FROM watch_spy').fetchone()),('','exempt'))
        self.c.handle(500,2,500,'/ignore del -10003')
        self.assertFalse(excluded(self.c,-10003))
        self.c.handle(500,3,500,'/ignore del -10002')
        self.assertTrue(excluded(self.c,-10002))

    def test_only_authorized_join_in_control_chats_and_revocation_cancels(self):
        self.c.handle(600,1,600,'/join @groupname')
        self.c.handle(-10003,1,500,'/join @groupname')
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM watch_join_jobs').fetchone()[0],0)
        self.c.handle(500,1,500,'/sudo 600')
        self.c.handle(600,2,600,'/join @groupname')
        self.c.handle(600,3,600,'/ignore -10003')
        self.assertFalse(excluded(self.c,-10003))
        self.c.handle(500,2,500,'/sudo del 600')
        transport=SimpleNamespace(join_group=AsyncMock())
        asyncio.run(joins(self.c,transport))
        transport.join_group.assert_not_called()

    def test_join_replay_and_restart_never_repeat_rpc(self):
        self.c.handle(500,1,500,'/join https://t.me/+abcdefgh1234')
        self.c.handle(500,1,500,'/join https://t.me/+abcdefgh1234')
        transport=SimpleNamespace(join_group=AsyncMock(side_effect=TimeoutError))
        asyncio.run(joins(self.c,transport))
        interrupted(self.c)
        asyncio.run(joins(self.c,transport))
        self.assertEqual(transport.join_group.await_count,1)
        self.assertEqual(self.c.db.execute('SELECT target,state FROM watch_join_jobs').fetchone()[0],'')

    def test_private_join_request_enrolls_only_after_membership_is_confirmed(self):
        self.c.handle(500,1,500,'/join https://t.me/+abcdefgh1234')
        t=SimpleNamespace(join_group=AsyncMock(return_value=('等待群組管理員批准',True)),
                          joined_group=AsyncMock(return_value=-10004))
        asyncio.run(joins(self.c,t))
        self.assertTrue(excluded(self.c,-10004))
        with self.c.db:
            self.c.db.execute('UPDATE watch_join_jobs SET next_at=0')
        asyncio.run(joins(self.c,t))
        self.assertFalse(excluded(self.c,-10004))
        self.assertEqual(t.join_group.await_count,1)
        with self.c.db:
            self.c.db.execute('UPDATE spy_groups SET created_at=? WHERE chat_id=-10004',(int(time.time())+10,))
        capture(self.c,-10004,1,100,'before registration',int(time.time()))
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM watch_spy').fetchone()[0],0)

    def test_join_parser_accepts_no_arbitrary_fetches(self):
        for value in ['http://t.me/+abcdefgh','https://evil.test/+abcdefgh','@ab','@bot/path','https://t.me/+abc?x=1']:
            with self.assertRaises(ValueError): target(value)
        self.assertEqual(target('@groupname'),('username','groupname'))

    def test_spy_delivery_never_bypasses_unknown_outcomes(self):
        old=self.ledger.create(('ban',101,self.add(user=101)),now=int(time.time())-100)
        self.ledger.uncertain(old['id'])
        case=self.add(user=100)
        with self.assertRaises(StopBridge): claim(self.c,('ban',100,case))
        with self.c.db:
            self.c.db.execute("UPDATE operations SET error='receiver_failed',reply_id=42 WHERE id=?",(old['id'],))
        op=claim(self.c,('ban',100,case))
        self.assertNotEqual(op['id'],old['id'])
        with self.assertRaises(StopBridge): claim(self.c,('ban',102,self.add(user=102)))
        self.assertEqual(self.c.db.execute('SELECT state FROM operations WHERE id=?',(old['id'],)).fetchone()[0],'uncertain')

    def prepare_source(self):
        case=self.add(user=100,action='report_approved',status='approved_and_banned',actor=500,reason='WATCH_SPY')
        self.db.executescript('ALTER TABLE cases ADD COLUMN netban_eligible INTEGER DEFAULT 1; ALTER TABLE review_updates ADD COLUMN chat_id INTEGER; CREATE TABLE report_deliveries(case_id,state,last_error,attempts,review_message_id);')
        self.db.execute("INSERT INTO review_updates VALUES (?,'approve','report',500,-10001)",(case,));self.db.commit()
        with self.c.db:
            self.c.db.execute("INSERT INTO watch_spy(id,chat_id,message_id,user_id,text,received_at,status) VALUES (?,-10003,1,100,'spam',?,'review')",(case,int(time.time())))
        return case

    def test_approved_only_and_reversal_removes_only_confirmed_ownership(self):
        case=self.prepare_source()
        self.assertEqual(candidates(self.c),[('ban',100,case)])
        op=claim(self.c,('ban',100,case))
        self.ledger.sent(op['id'],1);self.ledger.acknowledge(op['id'],2)
        self.assertEqual(candidates(self.c),[])
        self.retire(case)
        self.assertEqual(candidates(self.c),[('unban',100,case)])

    def test_rejected_or_whitelisted_case_cannot_be_sent(self):
        case=self.prepare_source()
        self.db.execute('INSERT INTO global_whitelist VALUES (100)');self.db.commit()
        self.assertEqual(candidates(self.c),[])
        self.db.execute('DELETE FROM global_whitelist');self.db.execute("UPDATE review_updates SET decision='reject'");self.db.commit()
        self.assertEqual(candidates(self.c),[])

    def test_ack_is_observed_without_resending_after_restart(self):
        case=self.prepare_source()
        op=claim(self.c,('ban',100,case))
        t=SimpleNamespace(reconcile_operation=AsyncMock(),send=AsyncMock())
        asyncio.run(tick(self.c,t))
        t.send.assert_not_called()
        self.assertEqual(t.reconcile_operation.call_args[0][1]['id'],op['id'])
