//! Durable model updates and review decisions. SQLite is the source of truth;
//! publish a model snapshot only after the whole transaction commits.
use super::*;

pub(super) const OTHER_BAN_IN_CHAT: &str = "SELECT EXISTS(SELECT 1 FROM cases c WHERE c.id != ?1 AND c.target_user_id=?3
                 AND c.action IN ('auto_ban','spam_ban','report_approved','guest_bot_ban','guest_invoker_ban','project_ban')
                 AND c.status NOT IN ('reversal_pending','reversed')
                 AND (c.status NOT IN ('ban_failed','ban_pending') OR
                    (c.chat_id=?2 AND EXISTS(SELECT 1 FROM origin_ban_jobs o WHERE o.case_id=c.id AND o.outcome_unknown=1)))
                 AND (c.chat_id=?2 OR c.action='project_ban' OR EXISTS(
                    SELECT 1 FROM network_ban_targets n WHERE n.case_id=c.id AND n.chat_id=?2)
                    OR EXISTS(SELECT 1 FROM network_deliveries d WHERE d.case_id=c.id AND d.chat_id=?2 AND d.outcome_unknown=1)))";

pub(super) fn stable_probability(log_odds: f64) -> f64 {
    if log_odds >= 0.0 {
        1.0 / (1.0 + (-log_odds).exp())
    } else {
        let odds = log_odds.exp();
        odds / (1.0 + odds)
    }
}

pub(super) fn passes_threshold(score: f64, threshold: f64) -> bool {
    score.is_finite()
        && threshold.is_finite()
        && (0.0..=1.0).contains(&score)
        && (0.0..=1.0).contains(&threshold)
        && score >= threshold
}

pub(super) fn write_sample(
    tx: &rusqlite::Transaction<'_>,
    label: &str,
    text: &str,
    case_id: Option<&str>,
) -> Result<()> {
    anyhow::ensure!(matches!(label, "spam" | "ham"), "invalid training label");
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return Ok(());
    }
    if let Some(case_id) = case_id {
        // Existing rows also protect pre-migration approvals. No destructive
        // deduplication of historical data is necessary to deploy this guard.
        let mut stmt = tx.prepare("SELECT label, text FROM training_samples WHERE case_id = ?1")?;
        let rows = stmt.query_map(params![case_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut duplicate = false;
        for row in rows {
            let (old_label, old_text) = row?;
            anyhow::ensure!(
                old_label == label,
                "case already has a different training label; revert it first"
            );
            duplicate |= old_text == text;
        }
        if duplicate {
            return Ok(());
        }
    }
    tx.execute(
        "INSERT INTO training_samples (label, text, case_id, created_at) VALUES (?1, ?2, ?3, ?4)",
        params![label, text, case_id, Utc::now().to_rfc3339()],
    )?;
    let (spam, ham) = if label == "spam" { (1, 0) } else { (0, 1) };
    for token in tokens {
        tx.execute(
            "INSERT INTO word_frequencies (word, spam_count, ham_count) VALUES (?1, ?2, ?3)
             ON CONFLICT(word) DO UPDATE SET spam_count = spam_count + excluded.spam_count,
             ham_count = ham_count + excluded.ham_count",
            params![token, spam, ham],
        )?;
    }
    tx.execute_batch(
        "INSERT INTO model_meta (key, value) VALUES ('spam_docs', (SELECT COUNT(*) FROM training_samples WHERE label='spam'))
         ON CONFLICT(key) DO UPDATE SET value=excluded.value;
         INSERT INTO model_meta (key, value) VALUES ('ham_docs', (SELECT COUNT(*) FROM training_samples WHERE label='ham'))
         ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
    )?;
    Ok(())
}

impl Runtime {
    pub(super) async fn with_model_transaction<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&rusqlite::Transaction<'_>) -> Result<T> + Send + 'static,
    {
        let mut model = self.model.clone().lock_owned().await;
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            let value = f(&tx)?;
            let next = Self::load_model(&tx)?;
            tx.commit()?;
            // spawn_blocking continues after its caller is cancelled. Keep
            // the model lock and publish here, before another writer starts.
            *model = next;
            Ok(value)
        })
        .await
    }

    pub(super) async fn review_guard(&self, case_id: &str) -> tokio::sync::OwnedMutexGuard<()> {
        let lock = {
            let mut locks = self.review_locks.lock().await;
            locks.retain(|_, lock| lock.strong_count() > 0);
            let lock = locks
                .get(case_id)
                .and_then(std::sync::Weak::upgrade)
                .unwrap_or_else(|| Arc::new(Mutex::new(())));
            locks.insert(case_id.to_string(), Arc::downgrade(&lock));
            lock
        };
        lock.lock_owned().await
    }

    pub(super) fn migrate_v17_to_v18(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS training_reviews (
                case_id TEXT PRIMARY KEY,
                decision TEXT NOT NULL CHECK(decision IN ('approve', 'reject')),
                actor_id INTEGER NOT NULL,
                decided_at TEXT NOT NULL
             );
             PRAGMA user_version = 18;",
        )?;
        tx.commit()?;
        Ok(())
    }

    pub(super) async fn train_atomic(
        &self,
        label: &str,
        text: &str,
        case_id: Option<&str>,
    ) -> Result<()> {
        let label = label.to_string();
        let text = text.to_string();
        let case_id = case_id.map(str::to_string);
        self.with_model_transaction(move |tx| write_sample(tx, &label, &text, case_id.as_deref()))
            .await
    }

    /// The first decision wins, including rejection. Model changes and the
    /// decision are one transaction; API callbacks cannot label twice, even
    /// after a restart or if editing the Telegram keyboard fails.
    #[cfg(test)]
    pub(super) async fn decide_training_review(
        &self,
        case_id: &str,
        decision: &str,
        actor_id: i64,
    ) -> Result<bool> {
        self.decide_training_review_at(case_id, decision, actor_id, None)
            .await
    }

    pub(super) async fn decide_training_review_at(
        &self,
        case_id: &str,
        decision: &str,
        actor_id: i64,
        location: Option<(i64, i32)>,
    ) -> Result<bool> {
        anyhow::ensure!(
            matches!(decision, "approve" | "reject"),
            "invalid review decision"
        );
        let case_id = case_id.to_string();
        let decision = decision.to_string();
        let test_group_id = self.config.test_group_id;
        self.with_model_transaction(move |tx| {
            let (action, status, text): (String, String, String) = tx.query_row(
                "SELECT action, status, evidence_text FROM cases WHERE id=?1", params![case_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
            anyhow::ensure!(action == ActionKind::SpamBan.as_str() && !matches!(status.as_str(), "reversed" | "reversal_pending" | "ban_failed" | "ban_pending"), "case is no longer eligible for training review");
            let changed = tx.execute(
                "INSERT OR IGNORE INTO training_reviews (case_id, decision, actor_id, decided_at) VALUES (?1, ?2, ?3, ?4)",
                params![case_id, decision, actor_id, Utc::now().to_rfc3339()],
            )? != 0;
            if changed && decision == "approve" {
                write_sample(tx, "spam", &text, Some(&case_id))?;
                network_delivery::enqueue_network_ban(tx, &case_id, test_group_id)?;
            }
            if changed {
                if let Some((chat,message))=location {
                    tx.execute("INSERT INTO review_updates(case_id,kind,decision,chat_id,message_id,actor_id) VALUES (?1,'train',?2,?3,?4,?5)",
                        params![case_id,decision,chat,message,actor_id])?;
                }
            }
            Ok(changed)
        }).await
    }

    pub(super) async fn remove_network_ban_target(
        &self,
        case_id: &str,
        chat_id: i64,
    ) -> Result<()> {
        let case_id = case_id.to_string();
        self.with_conn(move |conn| {
            conn.execute(
                "DELETE FROM network_ban_targets WHERE case_id=?1 AND chat_id=?2",
                params![case_id, chat_id],
            )?;
            Ok(())
        })
        .await
    }

    pub(super) async fn has_other_ban_in_chat(
        &self,
        case_id: &str,
        chat_id: i64,
        user_id: i64,
    ) -> Result<bool> {
        let case_id = case_id.to_string();
        self.with_conn(move |conn| {
            Ok(conn.query_row(
                OTHER_BAN_IN_CHAT,
                params![case_id, chat_id, user_id],
                |row| row.get(0),
            )?)
        })
        .await
    }
}
