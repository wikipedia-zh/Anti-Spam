"""Read SPB decisions and keep a separate delivery ledger."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
import uuid

POLICY = "spb-hexdrill-v1-display4"
HEX_USERNAME = "hexdrill_bot"
MAX_ROWS = 100_000
PROTECTED = {5172206551, 777000, 1087968824, 136817688}


class StopBridge(Exception):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def case_uuid(value):
    try:
        return str(uuid.UUID(value)) == value
    except (ValueError, TypeError, AttributeError):
        return False


def read_source(path, protected=()):
    """A single read transaction includes decisions, exemptions and roles.

    Do not use immutable=1: production uses WAL and keeps changing.
    """
    path = Path(path).resolve(strict=True)
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=3)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA trusted_schema=OFF")
        conn.execute("BEGIN")
        schema = conn.execute("PRAGMA user_version").fetchone()[0]
        if schema not in (40, 41):
            raise StopBridge("Unsupported SPB schema; review before resuming")
        count, highwater = conn.execute("SELECT COUNT(*),COALESCE(MAX(rowid),0) FROM cases").fetchone()
        if count > MAX_ROWS:
            raise StopBridge("Source exceeds the reviewed size limit")
        excluded = PROTECTED | set(protected)
        for table in ("global_whitelist", "maintainers", "reviewers"):
            excluded.update(row[0] for row in conn.execute(f"SELECT user_id FROM {table}"))
        rows = conn.execute("""SELECT c.id,c.action,c.status,c.target_user_id,c.actor_user_id,
            c.model_score,c.matched_rule_id,c.matched_rule_pattern,c.log_message_id,c.created_at,
            r.decision AS review_decision,r.kind AS review_kind,r.actor_id AS reviewer,
            j.ban_done,j.outcome_unknown
            FROM cases c LEFT JOIN review_updates r ON r.case_id=c.id
            LEFT JOIN origin_ban_jobs j ON j.case_id=c.id
            WHERE c.action IN ('auto_ban','report_approved')
            AND c.status IN ('auto_banned','approved_and_banned','banned_delete_failed')
            ORDER BY c.id""").fetchall()
        cases, skipped = {}, {}
        for row in rows:
            r = dict(row)
            reason = None
            score = r["model_score"]
            displayed = f"{score:.4f}" if isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1 else None
            if r["action"] == "auto_ban":
                markers = (r["matched_rule_pattern"] or "").split("；")
                if any(re.fullmatch(r"REGEX@[1-9][0-9]*", marker) for marker in markers):
                    reason = "regex"
                elif displayed == "1.0000":
                    reason = "model_display_1"
            elif r["action"] == "report_approved" and isinstance(r["actor_user_id"], int) and r["actor_user_id"] > 0:
                if r["review_decision"] is None:
                    reason = "approved_report_legacy"
                elif (r["review_decision"], r["review_kind"], r["reviewer"]) == ("approve", "report", r["actor_user_id"]):
                    reason = "approved_report"
            if reason is None:
                continue
            user = r["target_user_id"]
            rejection = None
            if not isinstance(user, int) or not 0 < user < 2**52:
                rejection = "channel_or_invalid_id"
            elif user in excluded:
                rejection = "protected_or_whitelisted"
            elif not case_uuid(r["id"]):
                rejection = "invalid_case_id"
            elif r["ban_done"] is not None and (r["ban_done"] != 1 or r["outcome_unknown"] != 0):
                rejection = "ban_not_confirmed"
            if rejection:
                skipped[rejection] = skipped.get(rejection, 0) + 1
                continue
            cases[r["id"]] = {"case_id": r["id"], "user_id": user, "reason": reason,
                "display_score": displayed, "created_at": r["created_at"], "log_message_id": r["log_message_id"]}
        conn.rollback()
        return {"policy": POLICY, "source": str(path), "schema": schema, "source_count": count,
                "highwater": highwater, "read_at": int(time.time()), "cases": cases, "skipped": skipped}
    finally:
        conn.close()


def preview(snapshot, output):
    output = Path(output)
    for candidate in (output, output.with_suffix(".csv")):
        if candidate.resolve() == Path(snapshot["source"]).resolve() or (candidate.exists() and Path(snapshot["source"]).exists() and os.path.samefile(candidate, snapshot["source"])):
            raise StopBridge("Preview must not overwrite the source database")
    payload = {"policy": POLICY, "source": snapshot["source"], "cases": snapshot["cases"]}
    document = {"sha256": digest(payload), "payload": payload, "created_at": snapshot["read_at"], "skipped": snapshot["skipped"]}
    output.write_text(canonical(document) + "\n", encoding="utf-8")
    output.chmod(0o600)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["User ID", "Case ID", "Reason", "Displayed score", "Created", "Log"])
        for item in payload["cases"].values():
            # No names, evidence or arbitrary reasons enter a CSV or command.
            stamp = item["created_at"] or ""
            if stamp.startswith(("=", "+", "-", "@")):
                stamp = "'" + stamp
            log = item["log_message_id"]
            writer.writerow([item["user_id"], item["case_id"], item["reason"], item["display_score"], stamp,
                             f"https://t.me/SpamProtectionLogging/{log}" if isinstance(log, int) and log > 0 else ""])
    csv_path.chmod(0o600)
    return document["sha256"]


class Ledger:
    def __init__(self, path, source):
        self.path = Path(path).resolve()
        if self.path == Path(source).resolve() or (self.path.exists() and Path(source).exists() and os.path.samefile(self.path, source)):
            raise StopBridge("Ledger must not be the SPB database")
        self.conn = sqlite3.connect(self.path, timeout=3)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS seen(case_id TEXT PRIMARY KEY,approved INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS owned(user_id INTEGER PRIMARY KEY,active INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS operations(id TEXT PRIMARY KEY,user_id INTEGER NOT NULL,action TEXT NOT NULL,
                case_id TEXT NOT NULL,command TEXT NOT NULL,random_id INTEGER NOT NULL,state TEXT NOT NULL,
                created_at INTEGER NOT NULL,message_id INTEGER,reply_id INTEGER,error TEXT);
            CREATE TABLE IF NOT EXISTS approvals(sha256 TEXT PRIMARY KEY,approved_at INTEGER NOT NULL);
        """)
        self.path.chmod(0o600)

    def close(self):
        self.conn.close()

    def get(self, key, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        self.conn.execute("INSERT INTO meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, canonical(value)))

    def initialize(self, snapshot, binding):
        if self.get("binding") is not None:
            raise StopBridge("Already initialized")
        with self.conn:
            self.put("binding", binding)
            self.put("policy", POLICY)
            self.put("source", snapshot["source"])
            self.put("highwater", snapshot["highwater"])
            self.put("source_count", snapshot["source_count"])
            self.conn.executemany("INSERT INTO seen VALUES (?,0)", [(i,) for i in snapshot["cases"]])

    def check(self, snapshot, binding):
        if self.get("binding") != binding or self.get("policy") != POLICY or self.get("source") != snapshot["source"]:
            raise StopBridge("Not initialized for this account, destination and source")
        if snapshot["highwater"] < self.get("highwater", 0) or snapshot["source_count"] < self.get("source_count", 0):
            raise StopBridge("Source history shrank; check for a restore before resuming")

    def approve(self, path, expected, snapshot, binding):
        self.check(snapshot, binding)
        raw = Path(path).read_bytes()
        if len(raw) > 32 * 1024 * 1024:
            raise StopBridge("Preview too large")
        doc = json.loads(raw)
        payload = doc["payload"]
        if digest(payload) != expected or doc["sha256"] != expected or payload["policy"] != POLICY or payload["source"] != snapshot["source"]:
            raise StopBridge("Preview digest or source mismatch")
        # Approval applies only to unchanged cases in the reviewed file.
        valid = [key for key, value in payload["cases"].items() if snapshot["cases"].get(key) == value]
        with self.conn:
            self.conn.executemany("INSERT INTO seen VALUES (?,1) ON CONFLICT(case_id) DO UPDATE SET approved=1", [(i,) for i in valid])
            self.conn.execute("INSERT OR IGNORE INTO approvals VALUES (?,?)", (expected, int(time.time())))
        return len(valid)

    def plan(self, snapshot, binding):
        self.check(snapshot, binding)
        known = {r[0] for r in self.conn.execute("SELECT case_id FROM seen")}
        unseen = {k: v for k, v in snapshot["cases"].items() if k not in known}
        if len({v["user_id"] for v in unseen.values()}) > 20:
            raise StopBridge("More than 20 new targets; preview and approve this batch")
        with self.conn:
            # Old report records without a decision receipt always need review.
            self.conn.executemany("INSERT INTO seen VALUES (?,?)", [(k, int(v["reason"] != "approved_report_legacy")) for k, v in unseen.items()])
            self.put("highwater", snapshot["highwater"])
            self.put("source_count", snapshot["source_count"])
        approved = {r[0] for r in self.conn.execute("SELECT case_id FROM seen WHERE approved=1")}
        active_users = {v["user_id"] for v in snapshot["cases"].values()}
        candidates = {}
        for key, item in snapshot["cases"].items():
            if key in approved:
                candidates.setdefault(item["user_id"], key)
        owned = {r[0] for r in self.conn.execute("SELECT user_id FROM owned WHERE active=1")}
        changes = [("unban", uid, "revoked") for uid in sorted(owned - active_users)]
        changes += [("ban", uid, candidates[uid]) for uid in sorted(candidates.keys() - owned)]
        if len(owned - active_users) > 20:
            raise StopBridge("More than 20 removals; inspect source and ledger")
        return changes

    def pending(self):
        row = self.conn.execute("SELECT * FROM operations WHERE state NOT IN ('acked','cancelled','rejected') ORDER BY created_at,id LIMIT 1").fetchone()
        return dict(row) if row else None

    def create(self, change, now=None):
        now = int(time.time()) if now is None else now
        if self.pending():
            raise StopBridge("An earlier command is not confirmed")
        recent = self.conn.execute("SELECT created_at FROM operations WHERE created_at>? ORDER BY created_at DESC", (now - 86400,)).fetchall()
        if (recent and now - recent[0][0] < 20) or sum(r[0] > now - 3600 for r in recent) >= 30 or len(recent) >= 150 or now < self.get("not_before", 0):
            return None
        action, user, case = change
        if action not in ("ban", "unban") or not isinstance(user, int) or not 0 < user < 2**52 or (case != "revoked" and not case_uuid(case)):
            raise StopBridge("Invalid command target")
        op = str(uuid.uuid4())
        reference = uuid.UUID(op).hex[:12]
        command = f"/{action} {user} +spb SPB {case} sync:{reference}"
        with self.conn:
            self.conn.execute("INSERT INTO operations(id,user_id,action,case_id,command,random_id,state,created_at) VALUES (?,?,?,?,?,?,'sending',?)", (op, user, action, case, command, secrets.randbits(63) or 1, now))
        return self.pending()

    def sent(self, op, message_id):
        with self.conn:
            self.conn.execute("UPDATE operations SET state='sent',message_id=? WHERE id=? AND state IN ('sending','uncertain')", (message_id, op))

    def uncertain(self, op):
        with self.conn:
            self.conn.execute("UPDATE operations SET state='uncertain',error='result_unconfirmed' WHERE id=? AND state!='acked'", (op,))

    def flood(self, op, seconds):
        with self.conn:
            self.conn.execute("UPDATE operations SET state='cancelled',error='flood_wait' WHERE id=? AND state='sending'", (op,))
            self.put("not_before", int(time.time()) + max(1, seconds) + 5)

    def acknowledge(self, op, reply_id):
        row = self.conn.execute("SELECT * FROM operations WHERE id=?", (op,)).fetchone()
        if not row or row["state"] not in ("sent", "uncertain") or not row["message_id"]:
            raise StopBridge("No sent operation to acknowledge")
        with self.conn:
            self.conn.execute("UPDATE operations SET state='acked',reply_id=? WHERE id=?", (reply_id, op))
            self.conn.execute("INSERT INTO owned VALUES (?,?) ON CONFLICT(user_id) DO UPDATE SET active=excluded.active", (row["user_id"], int(row["action"] == "ban")))


def valid_ack(message, operation, hex_id, destination_id):
    if (message.get("chat_id") != destination_id or message.get("sender_id") != hex_id
            or message.get("reply_to") != operation["message_id"] or message.get("forwarded")
            or message.get("via_bot") or message.get("out")):
        return False
    lines = message.get("text", "").strip().splitlines()
    ending = "✅ +spb：已加入，封鎖（永久）" if operation["action"] == "ban" else "➖ +spb：已移除"
    return len(lines) >= 2 and lines[-2:] == ["📋 阻擋清單：", ending]
