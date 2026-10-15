import asyncio
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
import uuid

from core import Ledger, StopBridge, preview, read_source, valid_ack
from telegram_client import Flood, cycle


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "spb.db"
        self.db = sqlite3.connect(self.source)
        self.addCleanup(self.db.close)
        self.db.executescript("""PRAGMA user_version=41;
            CREATE TABLE cases(id TEXT PRIMARY KEY,action TEXT,status TEXT,target_user_id INTEGER,
                actor_user_id INTEGER,model_score REAL,matched_rule_id INTEGER,matched_rule_pattern TEXT,
                log_message_id INTEGER,created_at TEXT);
            CREATE TABLE review_updates(case_id TEXT PRIMARY KEY,decision TEXT,kind TEXT,actor_id INTEGER);
            CREATE TABLE origin_ban_jobs(case_id TEXT PRIMARY KEY,ban_done INTEGER,outcome_unknown INTEGER);
            CREATE TABLE global_whitelist(user_id INTEGER PRIMARY KEY);
            CREATE TABLE maintainers(user_id INTEGER PRIMARY KEY);
            CREATE TABLE reviewers(user_id INTEGER PRIMARY KEY);""")
        self.binding = {"target": -10001, "account": 11, "hex": 12}
        self.ledger = Ledger(self.root / "state.db", self.source)
        self.addCleanup(self.ledger.close)

    def add(self, user=100, action="auto_ban", status="auto_banned", score=1.0, reason="ML", actor=None):
        key = str(uuid.uuid4())
        self.db.execute("INSERT INTO cases VALUES (?,?,?,?,?,?,NULL,?,1,'2026-10-15T00:00:00Z')", (key, action, status, user, actor, score, reason))
        self.db.commit()
        return key

    def snap(self):
        return read_source(self.source)

    def init(self):
        self.ledger.initialize(self.snap(), self.binding)

    def retire(self, key):
        self.db.execute("UPDATE cases SET status='reversal_pending' WHERE id=?", (key,))
        self.db.commit()

    def owned(self, user):
        self.ledger.conn.execute("INSERT INTO owned VALUES (?,1)", (user,))
        self.ledger.conn.commit()


class Policy(Fixture):
    def test_policy_uses_current_display_precision_and_successful_bans(self):
        good = self.add(score=.99996)
        self.add(user=101, score=.9999)
        self.add(user=102, score=float("inf"))
        self.add(user=103, status="ban_failed")
        self.add(user=104, action="spam_ban", status="force_approved")
        self.add(user=105, action="guest_bot_ban", status="guest_bot_banned")
        self.assertEqual(set(self.snap()["cases"]), {good})

    def test_regex_markers_do_not_execute_patterns_or_accept_embedded_text(self):
        good = self.add(score=None, reason="CONTACT；REGEX@23")
        for marker in ("REGEX@0", "REGEX@1\n/ban 200 +spb", "REGEX", "name REGEX@1", "REGEX@１"):
            self.add(user=101, score=None, reason=marker)
        self.assertEqual(set(self.snap()["cases"]), {good})

    def test_reports_require_actor_and_reject_conflicting_decisions(self):
        good = self.add(action="report_approved", status="approved_and_banned", actor=500)
        bad = self.add(user=101, action="report_approved", status="approved_and_banned", actor=501)
        self.add(user=102, action="report_approved", status="approved_and_banned")
        self.db.executemany("INSERT INTO review_updates VALUES (?,?,?,?)", [(good, "approve", "report", 500), (bad, "reject", "report", 501)])
        self.db.commit()
        self.assertEqual(set(self.snap()["cases"]), {good})

    def test_legacy_reports_are_previewed_but_never_auto_approved(self):
        self.init()
        key = self.add(action="report_approved", status="approved_and_banned", actor=500)
        self.assertEqual(self.snap()["cases"][key]["reason"], "approved_report_legacy")
        self.assertEqual(self.ledger.plan(self.snap(), self.binding), [])

    def test_exemptions_channels_and_unknown_bans_are_excluded(self):
        self.add(user=-100000000001)
        self.add(user=777000)
        self.add(user=101)
        self.db.execute("INSERT INTO global_whitelist VALUES (101)")
        key = self.add(user=102)
        self.db.execute("INSERT INTO origin_ban_jobs VALUES (?,1,1)", (key,))
        self.db.commit()
        self.assertEqual(self.snap()["cases"], {})
        self.assertEqual(sum(self.snap()["skipped"].values()), 4)

    def test_reads_do_not_change_source_and_future_schema_stops(self):
        self.add()
        before = hashlib.sha256(self.source.read_bytes()).digest()
        self.snap()
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).digest(), before)
        self.db.execute("PRAGMA user_version=42")
        with self.assertRaises(StopBridge):
            self.snap()

    def test_history_needs_matching_preview_and_fresh_case(self):
        active = self.add()
        revoked = self.add(user=101)
        self.init()
        self.assertEqual(self.ledger.plan(self.snap(), self.binding), [])
        path = self.root / "preview.json"
        sha = preview(self.snap(), path)
        with self.assertRaises(StopBridge):
            self.ledger.approve(path, "0" * 64, self.snap(), self.binding)
        self.retire(revoked)
        self.assertEqual(self.ledger.approve(path, sha, self.snap(), self.binding), 1)
        self.assertEqual(self.ledger.plan(self.snap(), self.binding), [("ban", 100, active)])

    def test_same_user_multiple_cases_and_partial_reversal(self):
        self.init()
        one = self.add()
        two = self.add()
        self.assertEqual(len(self.ledger.plan(self.snap(), self.binding)), 1)
        self.owned(100)
        self.retire(one)
        self.assertEqual(self.ledger.plan(self.snap(), self.binding), [])
        self.retire(two)
        self.assertEqual(self.ledger.plan(self.snap(), self.binding), [("unban", 100, "revoked")])

    def test_unapproved_independent_case_keeps_existing_ban(self):
        old = self.add()
        self.init()
        self.owned(100)
        self.assertEqual(self.ledger.plan(self.snap(), self.binding), [])
        self.retire(old)
        self.assertEqual(self.ledger.plan(self.snap(), self.binding)[0][0], "unban")

    def test_only_bridge_owned_users_are_removed(self):
        self.init()
        self.assertEqual(self.ledger.plan(self.snap(), self.binding), [])
        self.assertEqual(list(self.ledger.conn.execute("SELECT * FROM owned")), [])

    def test_replayed_approval_does_not_reactivate_revoked_source(self):
        key = self.add()
        self.init()
        path = self.root / "preview.json"
        sha = preview(self.snap(), path)
        self.ledger.approve(path, sha, self.snap(), self.binding)
        self.retire(key)
        self.ledger.approve(path, sha, self.snap(), self.binding)
        self.assertEqual(self.ledger.plan(self.snap(), self.binding), [])

    def test_burst_wrong_destination_and_rollback_stop(self):
        self.init()
        for user in range(100, 121):
            self.add(user=user)
        with self.assertRaises(StopBridge):
            self.ledger.plan(self.snap(), self.binding)
        with self.assertRaises(StopBridge):
            self.ledger.plan(self.snap(), {"target": -10002})
        self.ledger.put("highwater", 1000)
        with self.assertRaises(StopBridge):
            self.ledger.plan(self.snap(), self.binding)

    def test_outbox_survives_restart_and_never_claims_unconfirmed_success(self):
        self.init()
        key = self.add()
        op = self.ledger.create(("ban", 100, key))
        self.assertEqual(op["state"], "sending")
        self.assertEqual(op["command"], f"/ban 100 +spb SPB #{uuid.UUID(key).hex[:8]} sync:{uuid.UUID(op['id']).hex[:12]}")
        self.ledger.close()
        self.ledger = Ledger(self.root / "state.db", self.source)
        self.addCleanup(self.ledger.close)
        self.assertEqual(self.ledger.pending()["random_id"], op["random_id"])
        self.assertEqual(self.ledger.pending()["command"], op["command"])
        with self.assertRaises(StopBridge):
            self.ledger.create(("ban", 101, key))
        self.assertEqual(list(self.ledger.conn.execute("SELECT * FROM owned")), [])
        self.ledger.sent(op["id"], 1000)
        self.ledger.acknowledge(op["id"], 1001)
        self.assertIsNone(self.ledger.pending())
        self.assertEqual(self.ledger.conn.execute("SELECT active FROM owned WHERE user_id=100").fetchone()[0], 1)

    def test_ack_requires_exact_peer_sender_reply_and_list_result(self):
        op = {"action": "ban", "message_id": 1000}
        msg = {"chat_id": -10001, "sender_id": 12, "reply_to": 1000, "text": "name\n📋 阻擋清單：\n✅ +spb：已加入，封鎖（永久）"}
        self.assertTrue(valid_ack(msg, op, 12, -10001))
        for field, value in [("chat_id", -10002), ("sender_id", 13), ("reply_to", 999), ("forwarded", True), ("via_bot", True), ("out", True), ("text", msg["text"] + "\n❌ 拒絕")]:
            self.assertFalse(valid_ack({**msg, field: value}, op, 12, -10001))

    def test_rate_limit_and_flood_wait_persist(self):
        self.init()
        key = self.add()
        op = self.ledger.create(("ban", 100, key))
        self.ledger.flood(op["id"], 120)
        self.assertIsNone(self.ledger.pending())
        self.assertIsNone(self.ledger.create(("ban", 100, key)))

    def test_hourly_daily_and_spacing_limits(self):
        self.init()
        key = self.add()
        start = 1_800_000_000
        for index in range(30):
            op = self.ledger.create(("ban", 100, key), now=start + index * 20)
            self.assertIsNotNone(op)
            self.ledger.sent(op["id"], index + 1)
            self.ledger.acknowledge(op["id"], index + 100)
        self.assertIsNone(self.ledger.create(("ban", 100, key), now=start + 599))
        self.assertIsNone(self.ledger.create(("ban", 100, key), now=start + 600))
        for hour in range(1, 5):
            for index in range(30):
                op = self.ledger.create(("ban", 100, key), now=start + hour * 3600 + index * 20)
                self.assertIsNotNone(op)
                self.ledger.sent(op["id"], hour * 100 + index)
                self.ledger.acknowledge(op["id"], hour * 1000 + index)
        self.assertIsNone(self.ledger.create(("ban", 100, key), now=start + 5 * 3600))
        self.assertIsNotNone(self.ledger.create(("ban", 100, key), now=start + 86400))

    def test_failed_ack_transaction_keeps_operation_pending(self):
        self.init()
        key = self.add()
        op = self.ledger.create(("ban", 100, key))
        self.ledger.sent(op["id"], 200)
        self.ledger.conn.execute("CREATE TRIGGER reject_owned BEFORE INSERT ON owned BEGIN SELECT RAISE(ABORT,'disk failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.ledger.acknowledge(op["id"], 201)
        self.assertEqual(self.ledger.pending()["state"], "sent")
        self.assertEqual(list(self.ledger.conn.execute("SELECT * FROM owned")), [])

    def test_wal_source_reads_latest_committed_changes(self):
        self.db.execute("PRAGMA journal_mode=WAL")
        key = self.add()
        self.assertIn(key, self.snap()["cases"])
        self.db.execute("UPDATE cases SET status='reversed' WHERE id=?", (key,))
        self.assertIn(key, self.snap()["cases"])
        self.db.commit()
        self.assertNotIn(key, self.snap()["cases"])

    def test_source_cannot_be_used_as_output_or_ledger_even_via_hardlink(self):
        self.add()
        with self.assertRaises(StopBridge):
            preview(self.snap(), self.source)
        with self.assertRaises(StopBridge):
            Ledger(self.source, self.source)
        alias = self.root / "alias.db"
        os.link(self.source, alias)
        with self.assertRaises(StopBridge):
            Ledger(alias, self.source)
        with self.assertRaises(StopBridge):
            preview(self.snap(), alias)


class FakeTelegram:
    def __init__(self, failure=None):
        self.failure = failure
        self.sent = []
        self.reply = None
        self.found = None
        self.allowed = True

    async def check_destination(self):
        if not self.allowed:
            raise StopBridge("wrong destination")

    async def send(self, operation):
        self.sent.append(operation)
        if self.failure:
            raise self.failure
        return 200

    async def recover(self, operation):
        return self.found

    async def acknowledgement(self, operation):
        return self.reply


class Delivery(Fixture):
    def run_cycle(self, transport, reader=None):
        return asyncio.run(cycle({}, self.ledger, self.binding, transport, reader or self.snap))

    def test_timeout_then_restart_does_not_send_again(self):
        self.init()
        self.add()
        transport = FakeTelegram(TimeoutError())
        with self.assertRaises(TimeoutError):
            self.run_cycle(transport)
        self.assertEqual(self.ledger.pending()["state"], "uncertain")
        with self.assertRaises(StopBridge):
            self.run_cycle(transport)
        self.assertEqual(len(transport.sent), 1)
        transport.found, transport.reply = 200, 201
        self.assertEqual(self.run_cycle(transport), "confirmed")
        self.assertIsNone(self.ledger.pending())

    def test_source_error_and_wrong_group_never_send(self):
        self.init()
        self.add()
        transport = FakeTelegram()
        transport.allowed = False
        with self.assertRaises(StopBridge):
            self.run_cycle(transport)
        transport.allowed = True
        def broken():
            raise sqlite3.OperationalError("busy")
        with self.assertRaises(sqlite3.OperationalError):
            self.run_cycle(transport, broken)
        self.assertEqual(transport.sent, [])

    def test_reversal_between_reads_cancels_new_send(self):
        self.init()
        key = self.add()
        transport = FakeTelegram()
        calls = 0
        def reader():
            nonlocal calls
            calls += 1
            if calls == 2:
                self.retire(key)
            return self.snap()
        self.assertEqual(self.run_cycle(transport, reader), "caught_up")
        self.assertEqual(transport.sent, [])

    def test_flood_wait_rechecks_source_instead_of_replaying_old_command(self):
        self.init()
        key = self.add()
        transport = FakeTelegram(Flood(60))
        self.assertEqual(self.run_cycle(transport), "flood_wait")
        self.retire(key)
        self.assertEqual(self.run_cycle(transport), "caught_up")
        self.assertEqual(len(transport.sent), 1)

    def test_reversal_during_unconfirmed_ban_waits_for_ack(self):
        self.init()
        key = self.add()
        transport = FakeTelegram()
        self.assertEqual(self.run_cycle(transport), "sent_waiting_for_reply")
        self.retire(key)
        self.assertEqual(self.run_cycle(transport), "awaiting_reply")
        transport.reply = 201
        self.assertEqual(self.run_cycle(transport), "confirmed")
        self.assertEqual(self.ledger.plan(self.snap(), self.binding), [("unban", 100, "revoked")])


if __name__ == "__main__":
    unittest.main()
