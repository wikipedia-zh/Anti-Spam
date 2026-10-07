use super::*;
use serde_json::{json, Value};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Query {
    pub view: String,
    #[serde(default)]
    pub search: String,
    #[serde(default)]
    pub offset: u32,
    #[serde(default)]
    pub filter: String,
}

impl Query {
    pub(super) fn valid(&self) -> bool {
        matches!(
            self.view.as_str(),
            "overview" | "cases" | "groups" | "people" | "audit" | "queue"
        ) && (self.filter.is_empty()
            || (self.view == "cases" && self.filter == "pending_review")
            || (self.view == "queue" && matches!(self.filter.as_str(), "failed" | "network")))
            && self.search.chars().count() <= 100
            && self.offset <= 100_000
    }
}

fn rows(conn: &Connection, sql: &str, values: impl rusqlite::Params) -> Result<Vec<Value>> {
    let mut stmt = conn.prepare(sql)?;
    let names = stmt
        .column_names()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let result = stmt.query_map(values, |row| {
        let mut object = serde_json::Map::new();
        for (index, name) in names.iter().enumerate() {
            let value = match row.get_ref(index)? {
                rusqlite::types::ValueRef::Null => Value::Null,
                rusqlite::types::ValueRef::Integer(n) => json!(n),
                rusqlite::types::ValueRef::Real(n) => json!(n),
                rusqlite::types::ValueRef::Text(s) => json!(String::from_utf8_lossy(s)),
                rusqlite::types::ValueRef::Blob(_) => Value::Null,
            };
            object.insert(name.clone(), value);
        }
        Ok(Value::Object(object))
    })?;
    Ok(result.collect::<rusqlite::Result<Vec<_>>>()?)
}

impl Runtime {
    pub(super) async fn host_query(&self, query: Query) -> Result<Value> {
        anyhow::ensure!(query.valid(), "invalid host query");
        let mut config = self.config.clone();
        config.hostctl_secret = self.hostctl_secret().await;
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            let now = Utc::now().timestamp();
            let search = query.search.trim();
            let id = search.parse::<i64>().ok();
            let params = params![search, id, query.offset];
            let mut items = match query.view.as_str() {
                "overview" => {
                    let work = queue_status::WORK;
                    let mut summary = rows(&tx, &format!("SELECT
                        (SELECT COUNT(*) FROM cases WHERE action='pending_report' AND status='pending_review') AS pending_reports,
                        (SELECT COUNT(*) FROM ({work})) AS pending_work,
                        (SELECT COUNT(*) FROM ({work}) WHERE last_error IS NOT NULL) AS failed_work,
                        (SELECT COUNT(*) FROM network_deliveries WHERE state='pending') AS pending_network,
                        (SELECT COUNT(*) FROM group_module_settings WHERE chat_id NOT IN (SELECT chat_id FROM banned_groups)) AS known_groups,
                        (SELECT COUNT(*) FROM spam_rules) AS rules,
                        (SELECT not_before FROM telegram_retry_state WHERE id=1) AS telegram_not_before"), [])?.remove(0);
                    summary["schema"] = json!(tx.query_row("PRAGMA user_version", [], |r| r.get::<_,i64>(0))?);
                    summary["version"] = json!(env!("GIT_HASH"));
                    summary["global_threshold"] = json!(Self::load_threshold(&tx)?.unwrap_or(config.spam_threshold));
                    vec![summary]
                },
                "cases" => rows(&tx, "SELECT id,action,chat_id,target_user_id,target_name,
                    status,model_score,matched_rule_pattern AS reason,netban_eligible,
                    evidence_text AS evidence,created_at,
                    (SELECT COUNT(*) FROM network_deliveries n WHERE n.case_id=c.id AND n.state='done') AS network_done,
                    (SELECT COUNT(*) FROM network_deliveries n WHERE n.case_id=c.id AND n.state='pending') AS network_pending
                    FROM cases c WHERE (?1='' OR id=?1 OR target_user_id=?2 OR chat_id=?2)
                    AND (?4='' OR (action='pending_report' AND status='pending_review'))
                    ORDER BY rowid DESC LIMIT 26 OFFSET ?3", params![search,id,query.offset,query.filter])?,
                "groups" => rows(&tx, "SELECT g.chat_id,title,last_seen,netban,
                    spam_threshold_override,settings_revision,
                    EXISTS(SELECT 1 FROM banned_groups b WHERE b.chat_id=g.chat_id) AS service_denied
                    FROM group_module_settings g WHERE (?1='' OR g.chat_id=?2 OR INSTR(LOWER(COALESCE(title,'')),LOWER(?1))>0)
                    ORDER BY g.chat_id LIMIT 26 OFFSET ?3", params)?,
                "people" => rows(&tx, "SELECT * FROM (
                    SELECT 'host' AS role,?4 AS user_id,NULL AS added_by,NULL AS created_at
                    UNION ALL SELECT 'maintainer',user_id,added_by,created_at FROM maintainers WHERE user_id!=?4
                    UNION ALL SELECT 'reviewer',user_id,added_by,created_at FROM reviewers WHERE user_id!=?4)
                    WHERE (?1='' OR user_id=?2) ORDER BY role,user_id LIMIT 26 OFFSET ?3",
                    params![search,id,query.offset,HOST_ID])?,
                "audit" => rows(&tx, "SELECT * FROM (
                    SELECT CAST(action_id AS TEXT) AS id,'command' AS source,actor_id AS actor_user_id,chat_id,
                        command AS action,summary AS detail,reverted,created_at FROM maintainer_actions
                    UNION ALL SELECT request_id,'settings',actor_user_id,chat_id,'settings',
                        changes_json,0,created_at FROM group_settings_audit)
                    WHERE (?1='' OR id=?1 OR actor_user_id=?2 OR chat_id=?2)
                    ORDER BY created_at DESC,source,id DESC LIMIT 26 OFFSET ?3", params)?,
                "queue" => rows(&tx, &format!("SELECT kind,case_id,chat_id,attempts,next_attempt_at,
                    last_error FROM ({})
                    WHERE (?1='' OR case_id=?1 OR chat_id=?2)
                    AND (?4='' OR (?4='failed' AND last_error IS NOT NULL) OR (?4='network' AND kind='跨群封禁'))
                    ORDER BY (last_error IS NOT NULL) DESC,attempts DESC,next_attempt_at,kind,case_id,chat_id LIMIT 26 OFFSET ?3", queue_status::WORK), params![search,id,query.offset,query.filter])?,
                _ => unreachable!(),
            };
            let more = items.len() > 25;
            items.truncate(25);
            // Old command logs may contain sensitive diagnostic text too.
            for item in &mut items {
                if let Some(object) = item.as_object_mut() {
                    for value in object.values_mut() {
                        if let Some(text) = value.as_str() {
                            *value = json!(notices::diagnostic(&config,text));
                        }
                    }
                }
            }
            tx.commit()?;
            Ok(json!({"items":items,"has_more":more,"offset":query.offset,"updated_at":now}))
        }).await
    }
}
