use super::*;
use rusqlite::{OptionalExtension, Transaction};

pub(super) fn reporter_state(tx: &Connection, reporter: Option<i64>) -> Result<(bool, i64)> {
    let Some(id) = reporter else {
        return Ok((true, 0));
    };
    let exempt = is_host(id)
        || tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM maintainers WHERE user_id=?1)",
            [id],
            |r| r.get::<_, bool>(0),
        )?;
    let count = tx
        .query_row(
            "SELECT rejected_count FROM report_offenses WHERE user_id=?1",
            [id],
            |r| r.get(0),
        )
        .optional()?
        .unwrap_or(0);
    Ok((exempt, count))
}

pub(super) fn report(
    tx: &Transaction<'_>,
    id: &str,
    decision: &str,
    actor: (i64, &str),
    location: Option<(i64, i32)>,
) -> Result<bool> {
    anyhow::ensure!(
        matches!(decision, "approve" | "reject"),
        "invalid report decision"
    );
    let pending: Option<(String,Option<i64>,bool)> = tx.query_row(
        "SELECT evidence_text,actor_user_id,source_message_id IS NULL FROM cases WHERE id=?1 AND action='pending_report' AND status='pending_review'",
        [id], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
    let Some((text, reporter, no_source)) = pending else {
        return Ok(false);
    };
    let mut note = String::new();
    if decision == "approve" {
        tx.execute("UPDATE cases SET action='report_approved',status='ban_pending',actor_user_id=?2,actor_name=?3,log_message_id=NULL WHERE id=?1", params![id,actor.0,actor.1])?;
        tx.execute("INSERT INTO origin_ban_jobs(case_id,header,delete_done) VALUES (?1,'<b>舉報已受理，對象已封禁</b>',?2)", params![id,no_source])?;
        tx.execute(
            "INSERT INTO ban_followups(case_id,training_mode,audit_done) VALUES (?1,'report',1)",
            [id],
        )?;
    } else {
        reliability::write_sample(tx, "ham", &text, Some(id))?;
        let (exempt, _) = reporter_state(tx, reporter)?;
        if let Some(reporter) = reporter.filter(|_| !exempt) {
            tx.execute("INSERT INTO report_offenses(user_id,rejected_count,last_rejected_at) VALUES (?1,1,?2)
                ON CONFLICT(user_id) DO UPDATE SET rejected_count=rejected_count+1,last_rejected_at=excluded.last_rejected_at", params![reporter,Utc::now().to_rfc3339()])?;
            let count: i64 = tx.query_row(
                "SELECT rejected_count FROM report_offenses WHERE user_id=?1",
                [reporter],
                |r| r.get(0),
            )?;
            note = if count >= REPORT_STRIKE_LIMIT {
                format!("\n<b>舉報者</b>: <code>{reporter}</code> 已累計 {count} 次被拒，已暫停使用 /spam")
            } else {
                format!("\n<b>舉報者</b>: <code>{reporter}</code> 已累計 {count}/{REPORT_STRIKE_LIMIT} 次被拒")
            };
        }
        tx.execute("UPDATE cases SET action='report_rejected',status='rejected_and_cleaned',actor_user_id=?2,actor_name=?3 WHERE id=?1", params![id,actor.0,actor.1])?;
    }
    let (chat, message) = location.unzip();
    tx.execute("INSERT INTO review_updates(case_id,kind,decision,chat_id,message_id,confirmation_chat_id,confirmation_message_id,note,actor_id)
        SELECT ?1,'report',?2,?3,?4,(SELECT chat_id FROM report_confirmations WHERE case_id=?1),
        (SELECT message_id FROM report_confirmations WHERE case_id=?1),?5,?6", params![id,decision,chat,message,note,actor.0])?;
    tx.execute("DELETE FROM report_confirmations WHERE case_id=?1", [id])?;
    Ok(true)
}

pub(super) fn training(
    tx: &Transaction<'_>,
    id: &str,
    decision: &str,
    actor: i64,
    location: Option<(i64, i32)>,
    test_group: Option<i64>,
) -> Result<bool> {
    anyhow::ensure!(
        matches!(decision, "approve" | "reject"),
        "invalid review decision"
    );
    let (action, status, text): (String, String, String) = tx.query_row(
        "SELECT action,status,evidence_text FROM cases WHERE id=?1",
        [id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;
    anyhow::ensure!(
        action == ActionKind::SpamBan.as_str()
            && !matches!(
                status.as_str(),
                "reversed" | "reversal_pending" | "ban_failed" | "ban_pending"
            ),
        "case is no longer eligible for training review"
    );
    let changed=tx.execute("INSERT OR IGNORE INTO training_reviews(case_id,decision,actor_id,decided_at) VALUES (?1,?2,?3,?4)",params![id,decision,actor,Utc::now().to_rfc3339()])?!=0;
    if changed {
        if decision == "approve" {
            reliability::write_sample(tx, "spam", &text, Some(id))?;
            network_delivery::enqueue_network_ban(tx, id, test_group)?;
        }
        let (chat, message) = location.unzip();
        tx.execute("INSERT INTO review_updates(case_id,kind,decision,chat_id,message_id,actor_id) VALUES (?1,'train',?2,?3,?4,?5)",params![id,decision,chat,message,actor])?;
    }
    Ok(changed)
}

impl Runtime {
    pub(super) fn migrate_v31_to_v32(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch("CREATE TABLE review_updates_next (
            case_id TEXT PRIMARY KEY,kind TEXT NOT NULL,decision TEXT NOT NULL,actor_id INTEGER,
            chat_id INTEGER,message_id INTEGER,
            confirmation_chat_id INTEGER,confirmation_message_id INTEGER,
            note TEXT NOT NULL DEFAULT '',review_status TEXT NOT NULL DEFAULT '',
            confirmation_status TEXT NOT NULL DEFAULT '',attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt_at INTEGER NOT NULL DEFAULT 0,last_error TEXT);
            INSERT INTO review_updates_next(rowid,case_id,kind,decision,actor_id,chat_id,message_id,confirmation_chat_id,confirmation_message_id,note,review_status,confirmation_status,attempts,next_attempt_at,last_error)
                SELECT rowid,* FROM review_updates;
            DROP TABLE review_updates;
            ALTER TABLE review_updates_next RENAME TO review_updates;
            CREATE INDEX idx_review_update_due ON review_updates(next_attempt_at);
            CREATE TABLE IF NOT EXISTS training_review_locations(case_id TEXT PRIMARY KEY,chat_id INTEGER,message_id INTEGER);
            PRAGMA user_version=32;")?;
        let pending = {
            let mut stmt=tx.prepare("SELECT c.id,c.evidence_text FROM cases c LEFT JOIN ban_followups f ON f.case_id=c.id
                WHERE c.action='spam_ban' AND c.status NOT IN ('reversed','reversal_pending','ban_pending','ban_failed')
                AND COALESCE(c.matched_rule_pattern,'')!='BOTSPAM'
                AND (f.training_mode='review' OR (f.case_id IS NULL AND c.netban_eligible=0
                    AND NOT EXISTS(SELECT 1 FROM training_samples s WHERE s.case_id=c.id)))
                AND NOT EXISTS(SELECT 1 FROM training_reviews r WHERE r.case_id=c.id)")?;
            let rows = stmt
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        };
        for (id, text) in pending {
            if !is_empty_ml_text(&text) {
                tx.execute(
                    "INSERT OR IGNORE INTO training_review_locations(case_id) VALUES (?1)",
                    [id],
                )?;
            }
        }
        tx.commit()?;
        Ok(())
    }
}
