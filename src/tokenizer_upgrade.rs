//! Restore the one-character words discarded by the previous tokenizer.
//! Add only the missing counts; keep existing frequencies and manual biases.
use super::*;

impl Runtime {
    pub(super) fn migrate_v40_to_v41(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        for (label, text) in model_rebuild::samples(&tx)? {
            let (spam, ham) = match label.as_str() {
                "spam" => (1, 0),
                "ham" => (0, 1),
                _ => continue,
            };
            for token in tokenize(&text)
                .into_iter()
                .filter(|t| t.chars().count() == 1)
            {
                tx.execute(
                    "INSERT INTO word_frequencies(word,spam_count,ham_count) VALUES (?1,?2,?3)
                     ON CONFLICT(word) DO UPDATE SET spam_count=spam_count+excluded.spam_count,
                     ham_count=ham_count+excluded.ham_count",
                    params![token, spam, ham],
                )?;
            }
        }

        // write_sample used to return success without storing text made only
        // of single-character words. Recover only explicit, completed human
        // decisions whose text could not have produced an old-model sample.
        // Do not revive reversed bans, pending work, ordinary /sb bans, or
        // missing samples with old-model tokens (which may have been purged).
        let skipped: Vec<(String, String, String, String)> = {
            let mut stmt = tx.prepare(
                "SELECT c.id,c.evidence_text,
                    CASE WHEN c.action='report_rejected' THEN 'ham' ELSE 'spam' END,
                    COALESCE((SELECT r.decided_at FROM training_reviews r WHERE r.case_id=c.id AND r.decision='approve'),c.created_at)
                 FROM cases c
                 WHERE NOT EXISTS(SELECT 1 FROM training_samples s WHERE s.case_id=c.id)
                   AND COALESCE(c.matched_rule_pattern,'') != 'BOTSPAM'
                   AND (
                     (c.action='spam_ban' AND c.status='force_approved')
                     OR (c.action='spam_ban' AND c.status='done' AND EXISTS(
                         SELECT 1 FROM training_reviews r WHERE r.case_id=c.id AND r.decision='approve'))
                     OR (c.action='report_approved' AND c.status='approved_and_banned')
                     OR (c.action='report_rejected' AND c.status='rejected_and_cleaned')
                   ) ORDER BY c.created_at,c.id",
            )?;
            let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        let mut recovered = 0;
        for (id, text, label, decided_at) in skipped {
            let tokens = tokenize(&text);
            if !tokens.is_empty() && tokens.iter().all(|t| t.chars().count() == 1) {
                reliability::write_sample(&tx, &label, &text, Some(&id))?;
                // Preserve the historical decision date and make snapshot
                // upgrade checks deterministic across repeated trials.
                tx.execute(
                    "UPDATE training_samples SET created_at=?2 WHERE case_id=?1",
                    params![id, decided_at],
                )?;
                recovered += 1;
            }
        }
        tx.execute_batch("PRAGMA user_version=41;")?;
        tx.commit()?;
        log::info!("Tokenizer upgrade: restored single-character word counts; recovered {recovered} skipped reviewed samples");
        Ok(())
    }
}
