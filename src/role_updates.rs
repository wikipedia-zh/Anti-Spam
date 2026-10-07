use super::*;
use rusqlite::OptionalExtension;

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(super) enum Role {
    Maintainer,
    Reviewer,
}
impl Role {
    fn table(self) -> &'static str {
        match self {
            Self::Maintainer => "maintainers",
            Self::Reviewer => "reviewers",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(super) struct Snapshot {
    pub user_id: i64,
    pub revision: i64,
    pub host: bool,
    pub maintainer: bool,
    pub reviewer: bool,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Patch {
    pub request_id: String,
    pub user_id: i64,
    pub expected_revision: i64,
    pub role: Role,
    pub enabled: bool,
}

pub(super) enum SaveResult {
    Saved(Snapshot),
    Conflict,
    Invalid,
    Forbidden,
}

pub(super) fn valid_user(user_id: i64) -> bool {
    user_id > 0 && user_id < (1_i64 << 52)
}

fn snapshot(conn: &Connection, user_id: i64) -> Result<Snapshot> {
    Ok(Snapshot {
        user_id,
        revision: conn.query_row("SELECT revision FROM role_revision WHERE id=1", [], |r| {
            r.get(0)
        })?,
        host: is_host(user_id),
        maintainer: is_host(user_id)
            || conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM maintainers WHERE user_id=?1)",
                [user_id],
                |r| r.get::<_, bool>(0),
            )?,
        reviewer: conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM reviewers WHERE user_id=?1)",
            [user_id],
            |r| r.get(0),
        )?,
    })
}

fn change(
    tx: &rusqlite::Transaction<'_>,
    role: Role,
    user_id: i64,
    enabled: bool,
    actor: Option<i64>,
) -> Result<()> {
    if enabled {
        tx.execute(
            &format!(
                "INSERT OR IGNORE INTO {}(user_id,added_by,created_at) VALUES (?1,?2,?3)",
                role.table()
            ),
            params![user_id, actor, Utc::now().to_rfc3339()],
        )?;
    } else {
        tx.execute(
            &format!("DELETE FROM {} WHERE user_id=?1", role.table()),
            [user_id],
        )?;
    }
    Ok(())
}

impl Runtime {
    pub(super) fn migrate_v29_to_v30(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch("CREATE TABLE IF NOT EXISTS role_revision(id INTEGER PRIMARY KEY CHECK(id=1),revision INTEGER NOT NULL DEFAULT 0);
            INSERT OR IGNORE INTO role_revision(id) VALUES (1);
            CREATE TABLE IF NOT EXISTS host_role_requests(request_id TEXT PRIMARY KEY,actor_id INTEGER NOT NULL,payload TEXT NOT NULL,result TEXT NOT NULL,action_id INTEGER NOT NULL);")?;
        for table in ["maintainers", "reviewers"] {
            for op in ["INSERT", "UPDATE", "DELETE"] {
                tx.execute_batch(&format!("CREATE TRIGGER IF NOT EXISTS {table}_revision_{op} AFTER {op} ON {table} BEGIN UPDATE role_revision SET revision=revision+1 WHERE id=1; END;"))?;
            }
        }
        tx.execute_batch("PRAGMA user_version=30;")?;
        tx.commit()?;
        Ok(())
    }

    async fn with_role_transaction<T: Send + 'static>(
        &self,
        action: impl FnOnce(&rusqlite::Transaction<'_>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let mut cache = self.maintainers.clone().write_owned().await;
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            let result = action(&tx)?;
            let updated = {
                let mut stmt = tx.prepare("SELECT user_id FROM maintainers")?;
                let rows = stmt.query_map([], |r| r.get::<_, i64>(0))?;
                rows.collect::<rusqlite::Result<std::collections::HashSet<_>>>()?
            };
            tx.commit()?;
            *cache = updated;
            Ok(result)
        })
        .await
    }

    pub(super) async fn set_maintainer(
        &self,
        user_id: i64,
        enabled: bool,
        added_by: Option<i64>,
    ) -> Result<()> {
        self.with_role_transaction(move |tx| {
            change(tx, Role::Maintainer, user_id, enabled, added_by)
        })
        .await
    }

    pub(super) async fn set_reviewer(
        &self,
        user_id: i64,
        enabled: bool,
        added_by: Option<i64>,
    ) -> Result<()> {
        self.with_role_transaction(move |tx| change(tx, Role::Reviewer, user_id, enabled, added_by))
            .await
    }

    pub(super) async fn role_snapshot(&self, user_id: i64) -> Result<Snapshot> {
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            let result = snapshot(&tx, user_id)?;
            tx.commit()?;
            Ok(result)
        })
        .await
    }

    pub(super) async fn save_host_role(&self, actor_id: i64, patch: Patch) -> Result<SaveResult> {
        if !is_host(actor_id) {
            return Ok(SaveResult::Forbidden);
        }
        if is_host(patch.user_id) {
            return Ok(SaveResult::Forbidden);
        }
        if !valid_user(patch.user_id)
            || patch.expected_revision < 0
            || Uuid::parse_str(&patch.request_id).is_err()
        {
            return Ok(SaveResult::Invalid);
        }
        self.with_role_transaction(move|tx| {
            let payload=serde_json::to_string(&patch)?;
            let previous=tx.query_row("SELECT actor_id,payload,result FROM host_role_requests WHERE request_id=?1",[&patch.request_id],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?))).optional()?;
            if let Some((actor,saved,result))=previous {
                return Ok(if actor==actor_id && saved==payload {SaveResult::Saved(serde_json::from_str(&result)?)}else{SaveResult::Conflict});
            }
            let before=snapshot(tx,patch.user_id)?;
            if before.revision!=patch.expected_revision {return Ok(SaveResult::Conflict);}
            let old_enabled=match patch.role {Role::Maintainer=>before.maintainer,Role::Reviewer=>before.reviewer};
            if old_enabled==patch.enabled {return Ok(SaveResult::Invalid);}
            change(tx,patch.role,patch.user_id,patch.enabled,Some(actor_id))?;
            let after=snapshot(tx,patch.user_id)?;
            let (command,undo)=match patch.role {
                Role::Maintainer=>("/maintainer",UndoData::Maintainer{user_id:patch.user_id,old_enabled}),
                Role::Reviewer=>("/reviewer",UndoData::Reviewer{user_id:patch.user_id,old_enabled}),
            };
            let summary=format!("user_id={} {}→{}",patch.user_id,old_enabled,patch.enabled);
            tx.execute("INSERT INTO maintainer_actions(actor_id,actor_name,chat_id,command,summary,undo_data,created_at) VALUES (?1,'項目主持人',NULL,?2,?3,?4,?5)",params![actor_id,command,summary,serde_json::to_string(&undo)?,Utc::now().to_rfc3339()])?;
            let action_id=tx.last_insert_rowid();
            tx.execute("INSERT INTO host_role_requests(request_id,actor_id,payload,result,action_id) VALUES (?1,?2,?3,?4,?5)",params![patch.request_id,actor_id,payload,serde_json::to_string(&after)?,action_id])?;
            Ok(SaveResult::Saved(after))
        }).await
    }
}
