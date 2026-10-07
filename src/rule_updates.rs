use super::*;

impl Runtime {
    async fn with_rule_transaction<T: Send + 'static>(
        &self,
        change: impl FnOnce(&rusqlite::Transaction<'_>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let mut cache = self.spam_rules.clone().write_owned().await;
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            let result = change(&tx)?;
            let rules = Self::load_spam_rules(&tx)?;
            tx.commit()?;
            // The blocking operation owns the guard through commit and publication.
            *cache = rules;
            Ok(result)
        })
        .await
    }

    pub(super) async fn refresh_spam_rules(&self) -> Result<()> {
        let mut cache = self.spam_rules.clone().write_owned().await;
        self.with_conn(move |conn| {
            *cache = Self::load_spam_rules(conn)?;
            Ok(())
        })
        .await
    }

    pub(super) async fn add_spam_rule(&self, pattern: &str, description: &str) -> Result<i64> {
        FancyRegex::new(pattern).context("invalid regex pattern")?;
        let pattern = pattern.to_string();
        let description = description.to_string();
        self.with_rule_transaction(move |tx| {
            tx.execute(
                "INSERT INTO spam_rules (pattern,description) VALUES (?1,?2)",
                params![pattern, description],
            )?;
            Ok(tx.last_insert_rowid())
        })
        .await
    }

    pub(super) async fn update_spam_rule_pattern(&self, id: i64, pattern: &str) -> Result<bool> {
        FancyRegex::new(pattern).context("invalid regex pattern")?;
        let pattern = pattern.to_string();
        self.with_rule_transaction(move |tx| {
            Ok(tx.execute(
                "UPDATE spam_rules SET pattern=?2 WHERE id=?1",
                params![id, pattern],
            )? > 0)
        })
        .await
    }

    pub(super) async fn delete_spam_rule(&self, id: i64) -> Result<bool> {
        self.with_rule_transaction(move |tx| {
            Ok(tx.execute("DELETE FROM spam_rules WHERE id=?1", [id])? > 0)
        })
        .await
    }
}
