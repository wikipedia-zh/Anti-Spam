import time
from control import Control
from inbox import capture, outcomes, ROLE_REPLY
from test_bridge import Fixture


class Inbox(Fixture):
    def setUp(self):
        super().setUp()
        self.config = {"source_db": str(self.source), "operator_id": 500, "account_id": 11,
                       "hex_bot_id": 12, "destination_chat_id": -10001, "dm_enabled": True}
        self.c = Control(self.config, self.ledger)
        with self.c.db:
            self.c.advance(500,0)
            self.c.advance(-10001,0)

    def test_incoming_is_deduplicated_and_does_not_grant_access(self):
        now = int(time.time())
        capture(self.c, 600, 1, 'hello', now)
        capture(self.c, 600, 1, 'hello', now)
        capture(self.c, 600, 2, '/sudo 700', now)
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM watch_inbox').fetchone()[0], 1)
        self.assertFalse(self.c.authorized(600))
        self.assertFalse(self.c.authorized(700))
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM operations').fetchone()[0], 0)

    def test_normal_messages_get_one_full_name_reply_per_day(self):
        now = int(time.time())
        capture(self.c, 600, 1, 'hello', now)
        capture(self.c, 600, 2, 'hello again', now)
        with self.c.db:
            self.c.db.execute("UPDATE watch_inbox SET status='ham'")
        outcomes(self.c)
        outcomes(self.c)
        rows = self.c.db.execute('SELECT body FROM watch_outbox WHERE chat_id=600').fetchall()
        self.assertEqual([r[0] for r in rows], [ROLE_REPLY])
        self.assertIn('Spam Protection Bot', ROLE_REPLY)
        self.assertIn('https://t.me/SpamProtectionChat', ROLE_REPLY)

    def test_evidence_is_owner_private_only(self):
        capture(self.c, 600, 1, 'private original', int(time.time()))
        case = self.c.db.execute('SELECT id FROM watch_inbox').fetchone()[0]
        self.c.handle(500, 1, 500, '/sudo 700')
        with self.c.db:
            self.c.advance(700,0)
        self.c.handle(700, 1, 700, '/evidence #' + case[:8])
        self.c.handle(-10001, 1, 500, '/evidence #' + case[:8])
        self.c.handle(500, 2, 500, '/evidence #' + case[:8])
        rows = self.c.db.execute("SELECT chat_id,body FROM watch_outbox WHERE body LIKE '%private original%'").fetchall()
        self.assertEqual([r[0] for r in rows], [500])

    def test_history_and_service_accounts_never_enter_model_queue(self):
        now = int(time.time())
        for user,date in [(600,now-60),(777000,now),(11,now),(12,now)]:
            capture(self.c,user,1,'casino',date)
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM watch_inbox').fetchone()[0],0)
