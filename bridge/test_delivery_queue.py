import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

from bridge import binding
from control import Control
from core import valid_ack, valid_failure
from delivery_queue import changes, managed, state, status, tick
from telegram_client import Flood, Telegram
from test_bridge import Fixture


class Queue(Fixture):
    def setUp(self):
        super().setUp()
        self.config = {'source_db': str(self.source), 'operator_id': 500, 'account_id': 11,
                       'hex_bot_id': 12, 'destination_chat_id': -10001, 'allowed_observers': [500],
                       'receiver_scope_confirmed': True, 'owner_only_private': True, 'live': False}
        self.first = self.add()
        self.second = self.add(user=101)
        self.legacy = self.add(user=102, action='report_approved', status='approved_and_banned', actor=500)
        self.ledger.initialize(self.snap(), binding(self.config))
        with self.ledger.conn:
            self.ledger.conn.execute('UPDATE seen SET approved=1 WHERE case_id IN (?,?)', (self.first, self.second))
        self.c = Control(self.config, self.ledger)
        with self.c.db:
            self.c.advance(500, 0)
            self.c.advance(-10001, 0)
        self.t = SimpleNamespace(check_destination=AsyncMock(), confirmed_failure=AsyncMock(return_value=True),
                                 send=AsyncMock(return_value=50), reconcile_operation=AsyncMock())

    def failed(self, user=100, case=None):
        op = self.ledger.create(('ban', user, case or self.first), now=int(time.time()) - 200)
        self.ledger.sent(op['id'], 10)
        with self.c.db:
            self.c.db.execute("UPDATE operations SET state='uncertain',error='receiver_failed',reply_id=11 WHERE id=?", (op['id'],))
        return op

    def resume(self, message=1, **kwargs):
        self.c.handle(500, message, 500, '/queue resume', date=int(time.time()), **kwargs)

    def run_tick(self):
        asyncio.run(tick(self.c, self.t))

    def age(self):
        with self.c.db:
            self.c.db.execute('UPDATE operations SET created_at=created_at-60')

    def test_read_only_count_deduplicates_failed_and_excludes_unapproved(self):
        self.failed()
        before = self.c.db.total_changes
        self.assertIn('剩餘：2 筆', status(self.c))
        self.assertIn('已回覆失敗：1 筆', status(self.c))
        self.assertEqual(before, self.c.db.total_changes)
        self.assertEqual({x[1] for x in changes(self.c)}, {100, 101})

    def test_probe_success_drains_only_approved_and_replay_cannot_restart(self):
        old = self.failed()
        self.resume()
        self.run_tick()
        op = self.ledger.pending()
        self.assertEqual(op['user_id'], 100)
        self.assertEqual(state(self.c)['status'], 'probe')
        self.assertEqual(self.c.db.execute('SELECT state FROM operations WHERE id=?', (old['id'],)).fetchone()[0], 'cancelled')
        self.run_tick()
        self.assertEqual(self.t.send.await_count, 1)  # No batch before the receipt.
        self.ledger.acknowledge(op['id'], 51)
        self.run_tick()
        self.assertEqual(state(self.c)['status'], 'running')
        self.age()
        self.run_tick()
        second = self.ledger.pending()
        self.assertEqual(second['user_id'], 101)
        self.ledger.acknowledge(second['id'], 52)
        self.run_tick()
        self.run_tick()
        self.assertEqual(state(self.c)['status'], 'done')
        self.assertEqual(changes(self.c), [])
        self.resume()  # Same Telegram message is not a new authorization.
        self.assertEqual(state(self.c)['status'], 'done')
        self.assertEqual(self.c.db.execute('SELECT approved FROM seen WHERE case_id=?', (self.legacy,)).fetchone()[0], 0)

    def test_latest_failure_is_probed_even_if_another_user_sorts_first(self):
        self.failed(101, self.second)
        self.resume()
        self.run_tick()
        self.assertEqual(self.ledger.pending()['user_id'], 101)

    def test_repeated_resume_does_not_send_extra_probe(self):
        self.failed()
        self.resume()
        self.run_tick()
        self.resume(2)
        self.run_tick()
        self.assertEqual(self.t.send.await_count, 1)

    def test_new_failure_pauses_and_notifies_both_places(self):
        self.failed()
        self.resume()
        self.run_tick()
        op = self.ledger.pending()
        with self.c.db:
            self.c.db.execute("UPDATE operations SET state='uncertain',error='receiver_failed',reply_id=51 WHERE id=?", (op['id'],))
        self.run_tick()
        self.run_tick()
        self.assertEqual(state(self.c)['status'], 'paused')
        self.assertTrue(managed(self.c))
        self.assertEqual(self.t.send.await_count, 1)
        self.assertEqual({r[0] for r in self.c.db.execute("SELECT chat_id FROM watch_outbox WHERE key LIKE 'queue:%:paused:%'")}, {500, -10001})
        self.age()
        self.resume(2)
        self.run_tick()
        self.assertEqual(self.t.send.await_count, 2)

    def test_unknown_send_is_never_retried_even_on_resume_or_restart(self):
        self.failed()
        self.resume()
        self.t.send.side_effect = TimeoutError
        with self.assertRaises(TimeoutError):
            self.run_tick()
        self.assertEqual(state(self.c)['status'], 'paused')
        self.c = Control(self.config, self.ledger)
        self.resume(2)
        self.run_tick()
        self.assertEqual(self.t.send.await_count, 1)
        self.assertEqual(self.ledger.pending()['error'], 'result_unconfirmed')

    def test_restart_while_probe_in_flight_only_reconciles(self):
        self.failed()
        self.resume()
        self.run_tick()
        self.c = Control(self.config, self.ledger)
        self.run_tick()
        self.assertEqual(self.t.send.await_count, 1)
        self.t.reconcile_operation.assert_awaited()

    def test_revocation_before_retry_does_not_send_or_cancel_old_failure(self):
        old = self.failed()
        self.resume()
        self.retire(self.first)
        self.run_tick()
        self.t.send.assert_not_called()
        self.assertEqual(self.ledger.pending()['id'], old['id'])
        self.assertEqual(state(self.c)['status'], 'paused')

    def test_whitelist_change_during_destination_check_blocks_send(self):
        self.failed()
        self.resume()
        async def change():
            self.db.execute('INSERT INTO global_whitelist VALUES (100)')
            self.db.commit()
        self.t.check_destination.side_effect = change
        self.run_tick()
        self.t.send.assert_not_called()

    def test_forged_or_missing_failure_receipt_never_retries(self):
        old = self.failed()
        self.resume()
        self.t.confirmed_failure.return_value = False
        self.run_tick()
        self.t.send.assert_not_called()
        self.assertEqual(self.ledger.pending()['id'], old['id'])

    def test_flood_wait_does_not_unlock_batch(self):
        self.failed()
        self.resume()
        self.t.send.side_effect = Flood(120)
        with self.assertRaises(Flood):
            self.run_tick()
        self.run_tick()
        self.run_tick()
        self.assertEqual(state(self.c)['status'], 'probe')
        self.assertEqual(self.t.send.await_count, 1)

    def test_resume_requires_fresh_trusted_authorized_command(self):
        self.c.handle(-10001, 1, 600, '/queue resume', date=int(time.time()))
        self.resume(1, trusted=False)
        self.c.handle(500, 2, 500, '/queue resume', date=int(time.time()) - 121)
        self.assertEqual(state(self.c), {})
        with self.c.db:
            self.c.db.execute('INSERT INTO sudo_users VALUES (600,0)')
        self.c.handle(-10001, 2, 600, '/queue resume', date=int(time.time()))
        self.assertEqual(state(self.c)['status'], 'requested')
        with self.c.db:
            self.c.db.execute('DELETE FROM sudo_users')
        self.run_tick()
        self.t.send.assert_not_called()

    def test_reversal_of_confirmed_probe_is_removed_before_more_bans(self):
        self.failed()
        self.resume()
        self.run_tick()
        self.ledger.acknowledge(self.ledger.pending()['id'], 51)
        self.retire(self.first)
        self.run_tick()
        self.age()
        self.run_tick()
        self.assertEqual(self.ledger.pending()['action'], 'unban')

    def test_receipt_examples_and_wrong_list_or_sender(self):
        op = {'message_id': 10, 'action': 'ban'}
        message = {'chat_id': -10001, 'sender_id': 12, 'reply_to': 10,
                   'text': '🔨 星 已被封鎖（永久）。\n📝 理由：測試\n📋 阻擋清單：\n✅ +spb：已加入，封鎖（永久）'}
        self.assertTrue(valid_ack(message, op, 12, -10001))
        failure = dict(message, text='⚠️ 操作失敗，請稍後再試。')
        self.assertTrue(valid_failure(failure, op, 12, -10001))
        self.assertFalse(valid_ack(failure, op, 12, -10001))
        for field, value in [('sender_id', 600), ('chat_id', -10002), ('reply_to', 9),
                             ('forwarded', True), ('via_bot', True), ('out', True)]:
            self.assertFalse(valid_ack(dict(message, **{field: value}), op, 12, -10001))
            self.assertFalse(valid_failure(dict(failure, **{field: value}), op, 12, -10001))
        self.assertFalse(valid_ack(dict(message, text=message['text'].replace('+spb', '+other')), op, 12, -10001))

    def test_failure_revalidation_checks_exact_original_command(self):
        self.failed()
        op = self.ledger.pending()
        command = SimpleNamespace(chat_id=-10001, sender_id=11, out=True, raw_text=op['command'], fwd_from=None, edit_date=None)
        reply = SimpleNamespace(chat_id=-10001, sender_id=12, reply_to_msg_id=10,
                                raw_text='⚠️ 操作失敗，請稍後再試。', fwd_from=None, via_bot_id=None, out=False)
        client = SimpleNamespace(get_messages=AsyncMock(side_effect=[command, reply]))
        self.assertTrue(asyncio.run(Telegram(client, self.config).confirmed_failure(op)))
        command.raw_text = '/ban 999 +spb changed'
        client.get_messages = AsyncMock(return_value=command)
        self.assertFalse(asyncio.run(Telegram(client, self.config).confirmed_failure(op)))

    def test_full_speed_ignores_old_hourly_and_daily_caps_but_waits_for_ack(self):
        self.failed()
        with self.c.db:
            for i in range(150):
                self.c.db.execute("INSERT INTO operations(id,user_id,action,case_id,command,random_id,state,created_at) VALUES (?,100,'ban',?,'old',1,'cancelled',?)",
                                  (f'old-{i}', self.first, int(time.time())))
        self.resume()
        self.run_tick()
        self.assertEqual(self.t.send.await_count, 1)
        self.run_tick()
        self.assertEqual(self.t.send.await_count, 1)
        self.ledger.acknowledge(self.ledger.pending()['id'], 51)
        self.run_tick()
        self.assertEqual(self.t.send.await_count, 2)
        self.assertEqual(self.ledger.pending()['user_id'], 101)

    def test_failed_claim_rolls_back_cancellation(self):
        old = self.failed()
        self.resume()
        self.c.db.execute("CREATE TRIGGER fail_claim BEFORE INSERT ON operations BEGIN SELECT RAISE(ABORT,'test'); END")
        with self.assertRaises(Exception):
            self.run_tick()
        self.assertEqual(self.ledger.pending()['id'], old['id'])
        self.assertEqual(self.ledger.pending()['state'], 'uncertain')
        self.t.send.assert_not_called()

    def test_retries_reviewed_spy_case_and_preserves_reversal_link(self):
        self.config.update(spy_enabled=True, spb_bot_id=13)
        self.c = Control(self.config, self.ledger)
        case = self.add(user=103, action='report_approved', status='approved_and_banned', actor=500, reason='WATCH_SPY')
        self.db.executescript('ALTER TABLE cases ADD COLUMN netban_eligible INTEGER DEFAULT 1; ALTER TABLE review_updates ADD COLUMN chat_id INTEGER; CREATE TABLE report_deliveries(case_id,state,last_error,attempts,review_message_id);')
        self.db.execute("INSERT INTO review_updates VALUES (?,'approve','report',500,-10001)", (case,))
        self.db.commit()
        old = self.failed(103, case)
        with self.c.db:
            self.c.db.execute("INSERT INTO watch_spy(id,chat_id,message_id,user_id,text,received_at,status) VALUES (?,-10003,1,103,'spam',0,'review')", (case,))
            self.c.db.execute('INSERT INTO spy_operations VALUES (?,?)', (old['id'], case))
        self.assertIn('剩餘：3 筆', status(self.c))
        self.resume()
        self.run_tick()
        op = self.ledger.pending()
        self.assertEqual(op['user_id'], 103)
        self.assertEqual(self.c.db.execute('SELECT case_id FROM spy_operations WHERE operation_id=?', (op['id'],)).fetchone()[0], case)
