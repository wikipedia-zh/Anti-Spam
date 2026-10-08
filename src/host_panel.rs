use super::*;
use serde_json::{json, Value};

// Bounds are whole seconds. Remove fractions before SQLite parses them: its
// millisecond rounding would otherwise move 23:59:59.999999 into the next day.
const CREATED_SECONDS: &str = "CAST(strftime('%s',substr(created_at,1,19) || CASE
    WHEN substr(created_at,-1)='Z' THEN 'Z'
    WHEN substr(created_at,-6,1) IN ('+','-') THEN substr(created_at,-6)
    ELSE '' END) AS INTEGER)";

const PENDING_TRAINING: &str = "c.action='spam_ban' AND c.status NOT IN ('ban_pending','ban_failed','reversed','reversal_pending')
    AND EXISTS(SELECT 1 FROM training_review_locations l WHERE l.case_id=c.id)
    AND NOT EXISTS(SELECT 1 FROM training_reviews r WHERE r.case_id=c.id)";

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
    #[serde(default)]
    pub created_from: Option<i64>,
    #[serde(default)]
    pub created_before: Option<i64>,
}

impl Query {
    pub(super) fn valid(&self) -> bool {
        matches!(
            self.view.as_str(),
            "overview" | "cases" | "groups" | "people" | "audit" | "queue" | "rules"
        ) && (self.filter.is_empty()
            || (self.view == "cases"
                && matches!(
                    self.filter.as_str(),
                    "pending_review"
                        | "pending_training"
                        | "banned"
                        | "pending"
                        | "failed"
                        | "unconfirmed"
                        | "reversal_pending"
                        | "reversed"
                        | "rejected"
                        | "cancelled"
                ))
            || (self.view == "queue" && matches!(self.filter.as_str(), "failed" | "network"))
            || (self.view == "groups" && self.filter == "overrides"))
            && self.search.chars().count() <= 100
            && self.offset <= 100_000
            && ([self.created_from, self.created_before]
                .iter()
                .flatten()
                .all(|n| (0..=253_402_300_799).contains(n)))
            && (self.created_from.is_none() && self.created_before.is_none()
                || matches!(self.view.as_str(), "cases" | "audit"))
            && !matches!((self.created_from, self.created_before), (Some(from), Some(before)) if from >= before)
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
                        (SELECT COUNT(*) FROM cases c WHERE {PENDING_TRAINING}) AS pending_training,
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
                "cases" => rows(&tx, &format!("SELECT id,action,chat_id,target_user_id,target_name,
                    status,model_score,matched_rule_pattern AS reason,netban_eligible,
                    evidence_text AS evidence,created_at,
                    (SELECT COUNT(*) FROM network_deliveries n WHERE n.case_id=c.id AND n.state='done') AS network_done,
                    (SELECT COUNT(*) FROM network_deliveries n WHERE n.case_id=c.id AND n.state='pending') AS network_pending
                    FROM cases c WHERE (?1='' OR id=?1 OR target_user_id=?2 OR chat_id=?2)
                    AND (?4=''
                        OR (?4='pending_review' AND action='pending_report' AND status='pending_review')
                        OR (?4='pending_training' AND {PENDING_TRAINING})
                        OR (?4='banned' AND (status IN ('ban_done','auto_banned','guest_bot_banned','guest_invoker_banned','approved_and_banned','force_approved','banned_delete_failed')
                            OR (status='done' AND action IN ('spam_ban','project_ban'))))
                        OR (?4='pending' AND status IN ('ban_pending','action_pending'))
                        OR (?4='failed' AND status IN ('ban_failed','action_failed','banned_delete_failed'))
                        OR (?4='unconfirmed' AND status='action_unconfirmed')
                        OR (?4 IN ('reversal_pending','reversed') AND status=?4)
                        OR (?4='rejected' AND action='report_rejected')
                        OR (?4='cancelled' AND status='action_cancelled'))
                    AND (?5 IS NULL OR {CREATED_SECONDS}>=?5)
                    AND (?6 IS NULL OR {CREATED_SECONDS}<?6)
                    ORDER BY rowid DESC LIMIT 26 OFFSET ?3"), params![search,id,query.offset,query.filter,query.created_from,query.created_before])?,
                "rules" => rows(&tx, "SELECT r.id,r.description,SUBSTR(r.pattern,1,600) AS pattern,
                    (SELECT COUNT(*) FROM cases c WHERE c.matched_rule_id=r.id OR INSTR('；'||COALESCE(c.matched_rule_pattern,'')||'；','；REGEX@'||r.id||'；')>0) AS recorded_hits
                    FROM spam_rules r WHERE (?1='' OR r.id=?2 OR INSTR(LOWER(r.pattern),LOWER(?1))>0 OR INSTR(LOWER(r.description),LOWER(?1))>0)
                    ORDER BY r.id DESC LIMIT 26 OFFSET ?3",params)?,
                "groups" => rows(&tx, "SELECT g.chat_id,title,last_seen,netban,
                    spam_threshold_override,settings_revision,
                    EXISTS(SELECT 1 FROM banned_groups b WHERE b.chat_id=g.chat_id) AS service_denied,
                    (SELECT state FROM group_departures d WHERE d.chat_id=g.chat_id ORDER BY d.rowid DESC LIMIT 1) AS departure_state
                    FROM group_module_settings g WHERE (?1='' OR g.chat_id=?2 OR INSTR(LOWER(COALESCE(title,'')),LOWER(?1))>0)
                    AND (?4='' OR (spam_threshold_override IS NOT NULL AND chat_id NOT IN (SELECT chat_id FROM banned_groups)))
                    ORDER BY g.chat_id LIMIT 26 OFFSET ?3", params![search,id,query.offset,query.filter])?,
                "people" => rows(&tx, "SELECT * FROM (
                    SELECT 'host' AS role,?4 AS user_id,NULL AS added_by,NULL AS created_at
                    UNION ALL SELECT 'maintainer',user_id,added_by,created_at FROM maintainers WHERE user_id!=?4
                    UNION ALL SELECT 'reviewer',user_id,added_by,created_at FROM reviewers WHERE user_id!=?4)
                    WHERE (?1='' OR user_id=?2) ORDER BY role,user_id LIMIT 26 OFFSET ?3",
                    params![search,id,query.offset,HOST_ID])?,
                "audit" => rows(&tx, &format!("SELECT * FROM (
                    SELECT CAST(action_id AS TEXT) AS id,'command' AS source,actor_id AS actor_user_id,chat_id,
                        command AS action,summary AS detail,reverted,created_at FROM maintainer_actions
                    UNION ALL SELECT request_id,'settings',actor_user_id,chat_id,'settings',
                        changes_json,0,created_at FROM group_settings_audit)
                    WHERE (?1='' OR id=?1 OR actor_user_id=?2 OR chat_id=?2)
                    AND (?4 IS NULL OR {CREATED_SECONDS}>=?4)
                    AND (?5 IS NULL OR {CREATED_SECONDS}<?5)
                    ORDER BY created_at DESC,source,id DESC LIMIT 26 OFFSET ?3"), params![search,id,query.offset,query.created_from,query.created_before])?,
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
