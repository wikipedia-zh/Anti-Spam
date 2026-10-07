use super::*;
use rusqlite::OptionalExtension;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub(super) struct Snapshot {
    pub chat_id: i64,
    pub title: String,
    pub revision: i64,
    pub modules: BTreeMap<String, bool>,
    pub threshold_override: Option<f64>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Patch {
    pub request_id: String,
    pub expected_revision: i64,
    pub changes: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, PartialEq)]
pub(super) enum SaveResult {
    Saved(Snapshot),
    Conflict,
    Invalid,
    Forbidden,
}

pub(super) fn column(key: &str) -> Option<&'static str> {
    Some(match key {
        "nohalal" => "no_halal",
        "nosm" => "no_service_messages",
        "flood" => "flood_control",
        "captcha" => "captcha",
        "netban" => "netban",
        "cmdclean" => "cmd_clean",
        "guestban" => "guest_ban",
        "nocontact" => "no_contact",
        "novoice" => "no_voice",
        "noexec" => "no_exec",
        _ => return None,
    })
}

fn snapshot(conn: &Connection, chat_id: i64) -> Result<Snapshot> {
    let (title, revision, threshold_override) = conn.query_row(
        "SELECT COALESCE(title,''), settings_revision, spam_threshold_override FROM group_module_settings WHERE chat_id=?1",
        [chat_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ).optional()?.unwrap_or((String::new(), 0, None));
    let mut modules = BTreeMap::new();
    for (key, _, _) in PUBLIC_MODULES {
        let sql = format!(
            "SELECT {} FROM group_module_settings WHERE chat_id=?1",
            column(key).expect("public module")
        );
        let enabled: Option<bool> = conn.query_row(&sql, [chat_id], |r| r.get(0)).optional()?;
        modules.insert(
            key.to_string(),
            enabled.unwrap_or_else(|| module_flag(&GroupModuleSettings::default(), key).unwrap()),
        );
    }
    Ok(Snapshot {
        chat_id,
        title,
        revision,
        modules,
        threshold_override,
    })
}

impl Runtime {
    pub(super) fn migrate_v21_to_v22(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        Self::add_column_if_missing(
            &tx,
            "group_module_settings",
            "settings_revision",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        let columns: Vec<_> = PUBLIC_MODULES
            .iter()
            .map(|(key, _, _)| column(key).unwrap())
            .chain(["spam_threshold_override", "pol"])
            .collect();
        let changed = columns
            .iter()
            .map(|c| format!("NEW.{c} IS NOT OLD.{c}"))
            .collect::<Vec<_>>()
            .join(" OR ");
        tx.execute_batch(&format!(
            "CREATE TRIGGER IF NOT EXISTS group_settings_revision AFTER UPDATE OF {} ON group_module_settings
             WHEN {changed} BEGIN UPDATE group_module_settings SET settings_revision=OLD.settings_revision+1 WHERE chat_id=NEW.chat_id; END;",
            columns.join(",")
        ))?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS group_settings_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL UNIQUE,
            chat_id INTEGER NOT NULL,
            actor_user_id INTEGER NOT NULL,
            expected_revision INTEGER NOT NULL,
            changes_json TEXT NOT NULL,
            before_json TEXT NOT NULL,
            after_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        ); PRAGMA user_version=22;",
        )?;
        tx.commit()?;
        Ok(())
    }

    pub(super) async fn group_settings_snapshot(&self, chat_id: i64) -> Result<Snapshot> {
        self.with_conn(move |conn| snapshot(conn, chat_id)).await
    }

    pub(super) async fn save_group_settings(
        &self,
        chat_id: i64,
        user_id: i64,
        patch: Patch,
        can_edit_threshold: bool,
    ) -> Result<SaveResult> {
        if Uuid::parse_str(&patch.request_id).is_err()
            || patch.expected_revision < 0
            || patch.changes.is_empty()
            || patch.changes.len() > PUBLIC_MODULES.len() + 1
        {
            return Ok(SaveResult::Invalid);
        }
        for (key, value) in &patch.changes {
            if key == "threshold_override" {
                if !can_edit_threshold {
                    return Ok(SaveResult::Forbidden);
                }
                if !value.is_null()
                    && !value
                        .as_f64()
                        .is_some_and(|v| v.is_finite() && (0.50..=0.99).contains(&v))
                {
                    return Ok(SaveResult::Invalid);
                }
            } else if column(key).is_none() || !value.is_boolean() {
                return Ok(SaveResult::Invalid);
            }
        }
        // Readers and command writers use this same lock, so a stale cache
        // fill cannot land after a successful save invalidates it.
        let mut cache = self.group_module_cache.clone().write_owned().await;
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            let changes = serde_json::to_string(&patch.changes)?;
            let previous: Option<(i64,i64,i64,String,String)> = tx.query_row(
                "SELECT chat_id,actor_user_id,expected_revision,changes_json,after_json FROM group_settings_audit WHERE request_id=?1",
                [&patch.request_id], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?)),
            ).optional()?;
            if let Some((chat, actor, revision, old_changes, after)) = previous {
                return Ok(if chat == chat_id && actor == user_id && revision == patch.expected_revision && old_changes == changes {
                    SaveResult::Saved(serde_json::from_str(&after)?)
                } else { SaveResult::Conflict });
            }
            let before = snapshot(&tx, chat_id)?;
            if before.revision != patch.expected_revision { return Ok(SaveResult::Conflict); }
            let mut after = before.clone();
            for (key,value) in patch.changes {
                if key == "threshold_override" { after.threshold_override = value.as_f64(); }
                else { after.modules.insert(key, value.as_bool().unwrap()); }
            }
            tx.execute("INSERT OR IGNORE INTO group_module_settings(chat_id,no_contact) VALUES (?1,1)", [chat_id])?;
            let mut assignments = Vec::new();
            let mut values: Vec<rusqlite::types::Value> = vec![chat_id.into()];
            for (key, enabled) in &after.modules {
                assignments.push(format!("{}=?{}", column(key).unwrap(), values.len()+1));
                values.push(i64::from(*enabled).into());
            }
            assignments.push(format!("spam_threshold_override=?{}", values.len()+1));
            values.push(after.threshold_override.into());
            tx.execute(&format!("UPDATE group_module_settings SET {} WHERE chat_id=?1", assignments.join(",")), rusqlite::params_from_iter(values))?;
            let saved = snapshot(&tx, chat_id)?;
            tx.execute("INSERT INTO group_settings_audit(request_id,chat_id,actor_user_id,expected_revision,changes_json,before_json,after_json,created_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
                params![patch.request_id, chat_id, user_id, patch.expected_revision, changes, serde_json::to_string(&before)?, serde_json::to_string(&saved)?, Utc::now().to_rfc3339()])?;
            tx.commit()?;
            // The blocking task retains the lock even if an HTTP request is
            // cancelled while SQLite is committing.
            cache.remove(&chat_id);
            Ok(SaveResult::Saved(saved))
        }).await
    }
}
