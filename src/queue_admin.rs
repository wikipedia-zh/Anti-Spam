use super::*;
use rusqlite::{types::Value as SqlValue, OptionalExtension};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Target {
    Origin {
        case_id: String,
    },
    Network {
        case_id: String,
        chat_id: i64,
    },
    Catchup {
        chat_id: i64,
        message_id: i32,
        user_id: i64,
    },
    Departure {
        request_id: String,
    },
    Restriction {
        case_id: String,
    },
    Reversal {
        case_id: String,
    },
    Captcha {
        chat_id: i64,
        user_id: i64,
    },
    Report {
        case_id: String,
    },
    Warning {
        chat_id: i64,
        message_id: i32,
    },
    RuleNotice {
        id: String,
    },
    Review {
        case_id: String,
    },
}
impl Target {
    fn location(&self) -> (&'static str, &'static str, Vec<SqlValue>) {
        match self {
            Self::Origin { case_id } => (
                "origin_ban_jobs",
                "case_id=?1",
                vec![case_id.clone().into()],
            ),
            Self::Network { case_id, chat_id } => (
                "network_deliveries",
                "case_id=?1 AND chat_id=?2",
                vec![case_id.clone().into(), (*chat_id).into()],
            ),
            Self::Catchup {
                chat_id,
                message_id,
                user_id,
            } => (
                "network_catchups",
                "chat_id=?1 AND message_id=?2 AND user_id=?3",
                vec![(*chat_id).into(), (*message_id).into(), (*user_id).into()],
            ),
            Self::Departure { request_id } => (
                "group_departures",
                "request_id=?1",
                vec![request_id.clone().into()],
            ),
            Self::Restriction { case_id } => (
                "restriction_jobs",
                "case_id=?1",
                vec![case_id.clone().into()],
            ),
            Self::Reversal { case_id } => (
                "reversal_retries",
                "case_id=?1",
                vec![case_id.clone().into()],
            ),
            Self::Captcha { chat_id, user_id } => (
                "captcha_jobs",
                "chat_id=?1 AND user_id=?2",
                vec![(*chat_id).into(), (*user_id).into()],
            ),
            Self::Report { case_id } => (
                "report_deliveries",
                "case_id=?1",
                vec![case_id.clone().into()],
            ),
            Self::Warning {
                chat_id,
                message_id,
            } => (
                "warning_requests",
                "chat_id=?1 AND message_id=?2",
                vec![(*chat_id).into(), (*message_id).into()],
            ),
            Self::RuleNotice { id } => ("rule_notice_jobs", "id=?1", vec![id.clone().into()]),
            Self::Review { case_id } => {
                ("review_updates", "case_id=?1", vec![case_id.clone().into()])
            }
        }
    }
    fn valid(&self) -> bool {
        let (_, _, values) = self.location();
        values.iter().all(|v| match v {
            SqlValue::Text(s) => !s.is_empty() && s.len() <= 256,
            SqlValue::Integer(n) => *n != 0,
            _ => false,
        })
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Read {
    pub target: Target,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Retry {
    pub request_id: String,
    pub target: Target,
    pub expected_revision: String,
}
pub(super) enum Outcome {
    Ready(Value),
    Forbidden,
    Invalid,
    Missing,
    Conflict,
    Busy,
    Held,
}

fn row(
    conn: &Connection,
    table: &str,
    filter: &str,
    values: Vec<SqlValue>,
) -> Result<Option<Value>> {
    let mut stmt = conn.prepare(&format!("SELECT * FROM {table} WHERE {filter}"))?;
    let names = stmt
        .column_names()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    Ok(stmt
        .query_row(rusqlite::params_from_iter(values), |r| {
            let mut out = serde_json::Map::new();
            for (i, n) in names.iter().enumerate() {
                let v = match r.get_ref(i)? {
                    rusqlite::types::ValueRef::Null => Value::Null,
                    rusqlite::types::ValueRef::Integer(v) => json!(v),
                    rusqlite::types::ValueRef::Real(v) => json!(v),
                    rusqlite::types::ValueRef::Text(v) => json!(String::from_utf8_lossy(v)),
                    rusqlite::types::ValueRef::Blob(_) => Value::Null,
                };
                out.insert(n.clone(), v);
            }
            Ok(Value::Object(out))
        })
        .optional()?)
}
fn snapshot(conn: &Connection, target: &Target, config: &Config) -> Result<Option<Value>> {
    let key = serde_json::to_string(target)?;
    let Some(mut summary) = row(
        conn,
        &format!("({})", queue_status::WORK),
        "job_key=?1",
        vec![key.into()],
    )?
    else {
        return Ok(None);
    };
    let (table, filter, values) = target.location();
    let stored = row(conn, table, filter, values)?.context("missing queue record")?;
    let case = match summary["case_id"].as_str() {
        Some(id) => row(conn, "cases", "id=?1", vec![id.to_string().into()])?,
        None => None,
    };
    let controls = operations::controls(conn)?;
    let chat = summary["chat_id"].as_i64().context("missing queue chat")?;
    let access = group_access::snapshot(conn, chat)?;
    let waiting_for_group = matches!(target, Target::Network { .. } | Target::Catchup { .. })
        && matches!(access.0.as_str(), "left" | "unavailable");
    let dependency = if matches!(target, Target::Catchup { .. }) {
        row(
            conn,
            "network_deliveries",
            "case_id=?1 AND chat_id=?2",
            vec![
                stored["case_id"]
                    .as_str()
                    .context("missing catch-up case")?
                    .to_string()
                    .into(),
                chat.into(),
            ],
        )?
    } else {
        None
    };
    let waiting_for_dependency = dependency.as_ref().is_some_and(|d| d["state"] == "pending")
        && stored["notice_id"].is_null()
        && network_delivery::eligible_target(
            conn,
            stored["case_id"].as_str().unwrap_or_default(),
            chat,
            config.test_group_id,
        )?
        .is_some();
    let cooldown: i64 = conn.query_row(
        "SELECT not_before FROM telegram_retry_state WHERE id=1",
        [],
        |r| r.get(0),
    )?;
    let raw_payload = stored["payload"]
        .as_str()
        .and_then(|p| serde_json::from_str::<Value>(p).ok());
    let automatic = case
        .as_ref()
        .and_then(|c| c["action"].as_str())
        .is_some_and(|a| {
            matches!(
                a,
                "auto_ban"
                    | "guest_bot_ban"
                    | "guest_invoker_ban"
                    | "flood_mute"
                    | "cmd_clean_mute"
            )
        });
    let paused = match target {
        Target::Network { .. } => controls.network_paused,
        Target::Origin { .. } => {
            automatic && controls.automatic_pending_paused && stored["ban_done"] == 0
        }
        Target::Restriction { .. } => {
            automatic
                && controls.automatic_pending_paused
                && raw_payload
                    .as_ref()
                    .is_some_and(|p| p["step"] == "apply" && p["uncertain"] != true)
        }
        _ => false,
    };
    summary["revision"] = json!(format!(
        "{:x}",
        Sha256::digest(
            json!([stored, case, controls, cooldown, access, dependency])
                .to_string()
                .as_bytes()
        )
    ));
    summary["target"] = serde_json::to_value(target)?;
    summary.as_object_mut().unwrap().remove("job_key");
    summary["membership_state"] = json!(access.0);
    summary["waiting_for_group"] = json!(waiting_for_group);
    summary["waiting_for_dependency"] = json!(waiting_for_dependency);
    summary["paused"] = json!(paused);
    summary["telegram_not_before"] = json!(cooldown);
    summary["retryable"] = json!(
        !paused
            && !waiting_for_group
            && !waiting_for_dependency
            && summary["last_error"]
                .as_str()
                .is_some_and(|e| !e.is_empty())
    );
    if let Some(error) = summary["last_error"].as_str() {
        summary["last_error"] = json!(notices::diagnostic(config, error));
    }
    summary["updated_at"] = json!(Utc::now().timestamp());
    Ok(Some(summary))
}
fn receipt(conn: &Connection, actor: i64, patch: &Retry) -> Result<Option<Outcome>> {
    let old = conn
        .query_row(
            "SELECT actor_id,payload,result FROM queue_retry_requests WHERE request_id=?1",
            [&patch.request_id],
            |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            },
        )
        .optional()?;
    Ok(match old {
        Some((owner, payload, result)) => Some(
            if owner == actor && payload == serde_json::to_string(patch)? {
                Outcome::Ready(serde_json::from_str(&result)?)
            } else {
                Outcome::Conflict
            },
        ),
        None => None,
    })
}

impl Runtime {
    pub(super) fn migrate_v39_to_v40(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch("CREATE TABLE IF NOT EXISTS queue_retry_requests(request_id TEXT PRIMARY KEY,actor_id INTEGER NOT NULL,payload TEXT NOT NULL,target TEXT NOT NULL,requested_at INTEGER NOT NULL,result TEXT NOT NULL,action_id INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_queue_retry_target ON queue_retry_requests(target,requested_at);
            PRAGMA user_version=40;")?;
        tx.commit()?;
        Ok(())
    }
    pub(super) async fn host_queue_item(&self, actor: i64, target: Target) -> Result<Outcome> {
        if !is_host(actor) {
            return Ok(Outcome::Forbidden);
        }
        if !target.valid() {
            return Ok(Outcome::Invalid);
        }
        let mut config = self.config.clone();
        config.hostctl_secret = self.hostctl_secret().await;
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            let v = snapshot(&tx, &target, &config)?;
            tx.commit()?;
            Ok(v.map(Outcome::Ready).unwrap_or(Outcome::Missing))
        })
        .await
    }
    async fn queue_retry_guards(
        &self,
        target: &Target,
    ) -> Result<Vec<tokio::sync::OwnedMutexGuard<()>>> {
        let mut guards = Vec::new();
        match target {
            Target::Origin { case_id }
            | Target::Restriction { case_id }
            | Target::Reversal { case_id } => {
                guards.push(self.review_guard(case_id).await);
                if let Some(c) = self.load_case(case_id).await? {
                    guards.push(self.user_action_guard(c.target_user_id).await);
                }
            }
            Target::Network { case_id, .. } => {
                if let Some(c) = self.load_case(case_id).await? {
                    guards.push(self.user_action_guard(c.target_user_id).await);
                }
            }
            Target::Catchup { user_id, .. } | Target::Captcha { user_id, .. } => {
                guards.push(self.user_action_guard(*user_id).await)
            }
            Target::Departure { request_id } => {
                let id = request_id.clone();
                let chat = self
                    .with_conn(move |conn| {
                        Ok(conn
                            .query_row(
                                "SELECT chat_id FROM group_departures WHERE request_id=?1",
                                [id],
                                |r| r.get::<_, i64>(0),
                            )
                            .optional()?)
                    })
                    .await?;
                if let Some(chat) = chat {
                    guards.push(self.user_action_guard(chat).await);
                }
            }
            Target::Report { case_id } | Target::Review { case_id } => {
                guards.push(self.review_guard(case_id).await)
            }
            Target::Warning {
                chat_id,
                message_id,
            } => guards.push(
                self.review_guard(&format!("warning:{chat_id}:{message_id}"))
                    .await,
            ),
            Target::RuleNotice { id } => {
                guards.push(self.review_guard(&format!("rule-notice:{id}")).await)
            }
        }
        Ok(guards)
    }
    pub(super) async fn retry_host_queue(&self, actor: i64, patch: Retry) -> Result<Outcome> {
        if !is_host(actor) {
            return Ok(Outcome::Forbidden);
        }
        if !patch.target.valid()
            || Uuid::parse_str(&patch.request_id).is_err()
            || patch.expected_revision.len() != 64
            || !patch
                .expected_revision
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
        {
            return Ok(Outcome::Invalid);
        }
        let request = patch.clone();
        if let Some(old) = self
            .with_conn(move |conn| receipt(conn, actor, &request))
            .await?
        {
            return Ok(old);
        }
        let guards = match tokio::time::timeout(
            Duration::from_secs(2),
            self.queue_retry_guards(&patch.target),
        )
        .await
        {
            Ok(g) => g?,
            Err(_) => return Ok(Outcome::Busy),
        };
        let mut config = self.config.clone();
        config.hostctl_secret = self.hostctl_secret().await;
        self.with_conn(move|conn|{
            let _guards=guards;let tx=conn.transaction()?;
            if let Some(old)=receipt(&tx,actor,&patch)? {return Ok(old)}
            let Some(current)=snapshot(&tx,&patch.target,&config)? else {return Ok(Outcome::Missing)};
            if current["revision"]!=patch.expected_revision {return Ok(Outcome::Conflict)}
            if current["paused"]==true || current["waiting_for_group"]==true || current["waiting_for_dependency"]==true {return Ok(Outcome::Held)}
            if current["retryable"]!=true {return Ok(Outcome::Invalid)}
            let now=Utc::now().timestamp();let key=serde_json::to_string(&patch.target)?;
            if tx.query_row("SELECT EXISTS(SELECT 1 FROM queue_retry_requests WHERE target=?1 AND requested_at>?2-30)",params![key,now],|r|r.get::<_,bool>(0))? {return Ok(Outcome::Busy)}
            let due=now.max(current["telegram_not_before"].as_i64().context("missing cooldown")?);
            let (table,filter,mut values)=patch.target.location();let slot=values.len()+1;values.push(due.into());
            tx.execute(&format!("UPDATE {table} SET next_attempt_at=?{slot} WHERE {filter}"),rusqlite::params_from_iter(values))?;
            let summary=json!({"target":patch.target,"previous_due":current["next_attempt_at"],"next_attempt_at":due,"paused":current["paused"]});
            tx.execute("INSERT INTO maintainer_actions(actor_id,actor_name,chat_id,command,summary,undo_data,created_at) VALUES (?1,'項目主持人',?2,'重試工作',?3,?4,?5)",params![actor,current["chat_id"].as_i64(),summary.to_string(),serde_json::to_string(&UndoData::NotRevertible)?,Utc::now().to_rfc3339()])?;
            let action_id=tx.last_insert_rowid();
            let result=json!({"request_id":patch.request_id,"action_id":action_id,"next_attempt_at":due,"paused":current["paused"],"target":patch.target});
            tx.execute("INSERT INTO queue_retry_requests(request_id,actor_id,payload,target,requested_at,result,action_id) VALUES (?1,?2,?3,?4,?5,?6,?7)",params![patch.request_id,actor,serde_json::to_string(&patch)?,key,now,result.to_string(),action_id])?;
            tx.commit()?;Ok(Outcome::Ready(result))
        }).await
    }
}
