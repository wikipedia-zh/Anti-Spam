import asyncio
from datetime import datetime, timezone
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from control import Control, Watch
from log_import import parse, source, validate, capture, approved_cases
from spy_delivery import candidates, claim, tick
from delivery_queue import changes
from telegram_client import Telegram
from test_bridge import Fixture


LOGS = [
    '訪客模式機器人封禁 對象: 8803865976 GlenLeaf (@zxap6uobot) 群組: -1003518168907 案例: #d7f68ca9 原因: GUEST_MODE 2026-10-24 12:00:47 UTC+8\nspam evidence',
    '自動封禁\n對象: 8793875330 Heydo Sishdo\n群組: -1004356309107\n案例: #547399ca\n分數: 0.9994\n原因: ML\n2026-10-24 19:11:48 UTC+8\nexample spam',
    '受理封禁 對象: 6012880408 name (@Petterniker) 群組: -1002634625042 案例: #379f2a18 處理者: 8063256050 2026-10-24 20:54:30 UTC+8',
]


class Logs(Fixture):
    def setUp(self):
        super().setUp()
        self.db.executescript('ALTER TABLE cases ADD COLUMN chat_id INTEGER; ALTER TABLE cases ADD COLUMN netban_eligible INTEGER DEFAULT 1;')
        for text, action, status in zip(LOGS, ['guest_bot_ban', 'auto_ban', 'report_approved'],
                                         ['guest_bot_banned', 'auto_banned', 'approved_and_banned']):
            item = parse(text)
            self.db.execute("""INSERT INTO cases(id,action,status,target_user_id,chat_id,model_score,netban_eligible)
                VALUES (?,?,?,?,?,0.9994,1)""", (item['prefix'] + '-0000-4000-8000-000000000000', action, status, item['user_id'], item['chat_id']))
        self.db.commit()
        self.db.executescript('ALTER TABLE review_updates ADD COLUMN chat_id INTEGER; CREATE TABLE report_deliveries(case_id,state,last_error,attempts,review_message_id);')
        self.config = {'source_db': str(self.source), 'operator_id': 500, 'account_id': 11,
                       'hex_bot_id': 12, 'destination_chat_id': -10001, 'spy_enabled': True,
                       'spb_bot_id': 13, 'spy_excluded_chats': [], 'owner_only_private': True,
                       'receiver_scope_confirmed': True, 'allowed_observers': []}
        self.c = Control(self.config, self.ledger)
        from bridge import binding
        self.ledger.initialize(self.snap(), binding(self.config))

    def capture(self, index=0, message=1):
        return capture(self.c, 500, message, 500, LOGS[index], int(time.time()))

    def approve(self):
        with self.c.db:
            self.c.db.execute("UPDATE watch_log_requests SET status='approved',approved_by=500")

    def test_forward_only_requests_confirmation_and_deduplicates(self):
        for chat, actor in [(600, 600), (-10001, 500), (600, 500)]:
            self.assertFalse(capture(self.c, chat, 1, actor, LOGS[0], int(time.time())))
        self.assertTrue(self.capture())
        self.capture(message=2)
        self.capture(message=2)
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM watch_log_requests').fetchone()[0], 1)
        self.assertEqual(self.c.db.execute('SELECT status FROM watch_log_requests').fetchone()[0], 'pending')
        self.assertEqual(candidates(self.c), [])
        self.assertEqual(changes(self.c), [])
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM operations').fetchone()[0], 0)

    def test_approved_guest_and_subthreshold_log_survive_restart_and_reversal(self):
        for i in range(3): self.capture(i, i+1)
        self.approve()
        self.c = Control(self.config, self.ledger)
        todo = candidates(self.c)
        self.assertEqual({x[1] for x in todo}, {8803865976, 8793875330, 6012880408})
        self.assertEqual(set(changes(self.c)), set(todo))
        op = claim(self.c, next(x for x in todo if x[1] == 8803865976), now=int(time.time())-100)
        self.ledger.sent(op['id'], 100)
        self.ledger.acknowledge(op['id'], 101)
        self.assertNotIn(('unban', op['user_id'], op['case_id']), candidates(self.c))
        self.assertNotIn(('unban', op['user_id'], op['case_id']), changes(self.c))
        self.capture(message=5)
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM operations').fetchone()[0], 1)
        self.retire(op['case_id'])
        self.assertIn(('unban', op['user_id'], op['case_id']), candidates(self.c))
        self.assertIn(('unban', op['user_id'], op['case_id']), changes(self.c))

    def test_recheck_source_before_send_and_never_resend_unknown_result(self):
        self.capture(); self.approve()
        async def revoke():
            self.db.execute("UPDATE cases SET status='reversal_pending'"); self.db.commit()
        t = SimpleNamespace(check_destination=AsyncMock(side_effect=revoke), send=AsyncMock())
        asyncio.run(tick(self.c, t))
        t.send.assert_not_called()
        self.db.execute("UPDATE cases SET status='guest_bot_banned' WHERE action='guest_bot_ban'"); self.db.commit()
        op = claim(self.c, candidates(self.c)[0])
        self.ledger.uncertain(op['id'])
        t = SimpleNamespace(reconcile_operation=AsyncMock(), send=AsyncMock())
        asyncio.run(tick(self.c, t))
        t.send.assert_not_called()
        t.reconcile_operation.assert_awaited_once()
        self.db.execute('INSERT INTO global_whitelist VALUES (8803865976)'); self.db.commit()
        self.assertEqual(approved_cases(self.c), {})

    def test_rejection_and_explicit_card_retry_do_not_grant_approval(self):
        self.capture()
        with self.c.db:
            self.c.db.execute("UPDATE watch_log_requests SET status='rejected',review_state='sent',review_message=10,review_chat=-10001")
        self.assertEqual(candidates(self.c), [])
        self.capture(message=2)
        self.assertEqual(tuple(self.c.db.execute('SELECT status,review_state,review_message FROM watch_log_requests').fetchone()), ('pending','queued',None))
        with self.c.db:
            self.c.db.execute("UPDATE watch_log_requests SET review_state='uncertain'")
        self.capture(message=3)
        self.assertEqual(approved_cases(self.c), {})

    def test_transport_routes_owner_forward_but_never_forwarded_commands(self):
        class Client:
            def add_event_handler(self, handler, event): self.handler = handler
        client = Client()
        transport = Telegram(client, self.config)
        transport.send_notice = AsyncMock(return_value=90)
        watch = Watch(self.config, self.ledger, transport)
        event = SimpleNamespace(sender=None, input_chat='owner', message=SimpleNamespace(
            chat_id=500, sender_id=500, id=1, raw_text=LOGS[0], date=datetime.now(timezone.utc),
            fwd_from=object(), via_bot_id=None, out=False, edit_date=None, action=None))
        with patch.dict('sys.modules', {'telethon': SimpleNamespace(events=SimpleNamespace(NewMessage=lambda **kw: kw))}):
            transport.install_control_handler(watch.incoming)
        asyncio.run(client.handler(event))
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM watch_log_requests').fetchone()[0], 1)
        event.message.raw_text = '/sudo 600'; event.message.id = 2
        asyncio.run(client.handler(event))
        self.assertFalse(self.c.authorized(600))
        event.message.raw_text = LOGS[1]; event.message.id = 3; event.message.edit_date = datetime.now(timezone.utc)
        asyncio.run(client.handler(event))
        event.message.edit_date = None; event.message.chat_id = event.message.sender_id = 600
        asyncio.run(client.handler(event))
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM watch_log_requests').fetchone()[0], 1)

    def test_poll_recovers_missed_forward_once_and_skips_forwarded_commands(self):
        t = SimpleNamespace(control_messages=AsyncMock(return_value=[
            {'id': 1, 'actor': 500, 'text': LOGS[0], 'date': int(time.time()), 'trusted': False, 'log_trusted': True},
            {'id': 2, 'actor': 500, 'text': '/sudo 600', 'date': int(time.time()), 'trusted': False, 'log_trusted': True}]))
        watch = Watch(self.config, self.ledger, t)
        with self.c.db: self.c.advance(500, 0)
        asyncio.run(watch.poll(500, 500)); asyncio.run(watch.poll(500, 500))
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM watch_log_requests').fetchone()[0], 1)
        self.assertFalse(self.c.authorized(600))

    def test_three_log_formats_and_export_prefix(self):
        for text, user in zip(LOGS, [8803865976, 8793875330, 6012880408]):
            for prefix in ['', '[24/10/2026 12:00\u202fAM] Spam Protection Log: ']:
                item = parse(prefix + text)
                self.assertEqual(item['user_id'], user)
                db = source(self.source)
                try:
                    self.assertEqual(validate(db, item, set())['target_user_id'], user)
                    with self.assertRaises(Exception):
                        db.execute('UPDATE cases SET netban_eligible=0')
                finally:
                    db.close()

    def test_evidence_and_sender_name_cannot_inject_another_target(self):
        item = parse(LOGS[0] + '\n' + LOGS[1] + '\n/ban 123 +spb')
        self.assertEqual(item['user_id'], 8803865976)
        self.assertIsNone(parse(LOGS[0].replace('GlenLeaf', '對象: 123')))
        self.assertIsNone(parse('someone said ' + LOGS[0]))
        self.assertIsNone(parse(LOGS[0].replace('12:00:47', '99:00:00')))
        self.assertIsNone(parse(LOGS[0].replace('8803865976', '-100123')))
        self.assertIsNone(parse(LOGS[0].replace('#d7f68ca9', '#d7f68ca9dead')))

    def test_source_mismatch_revocation_protection_and_ambiguous_prefix(self):
        original = parse(LOGS[0])
        for item in [dict(original, user_id=123), dict(original, chat_id=-100123), dict(original, title='自動封禁')]:
            db = source(self.source)
            try:
                with self.assertRaises(ValueError): validate(db, item, set())
            finally:
                db.close()
        for sql in ["UPDATE cases SET status='reversal_pending'", "UPDATE cases SET netban_eligible=0",
                    'INSERT INTO global_whitelist VALUES (8803865976)',
                    "INSERT INTO origin_ban_jobs SELECT id,0,1 FROM cases",
                    "INSERT INTO cases(id) VALUES ('d7f68ca9-0000-4000-8000-000000000001')"]:
            self.db.execute('SAVEPOINT test')
            self.db.execute(sql)
            with self.assertRaises(ValueError):
                self.db.row_factory = __import__('sqlite3').Row
                validate(self.db, original, set())
            self.db.execute('ROLLBACK TO test')
            self.db.execute('RELEASE test')
        db = source(self.source)
        try:
            with self.assertRaises(ValueError): validate(db, original, {8803865976})
        finally:
            db.close()
