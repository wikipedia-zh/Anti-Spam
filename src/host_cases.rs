use super::*;
use rusqlite::OptionalExtension;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Query {
    pub case_id: String,
    #[serde(default)]
    pub offset: u32,
}
impl Query {
    pub fn valid(&self) -> bool {
        valid_id(&self.case_id) && self.offset <= 100_000
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Reverse {
    pub request_id: String,
    pub case_id: String,
    pub target_user_id: i64,
    pub expected_revision: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Review {
    pub request_id: String,
    pub case_id: String,
    pub target_user_id: i64,
    pub expected_revision: String,
    pub kind: String,
    pub decision: String,
}

#[derive(Debug)]
pub(super) enum Outcome {
    Saved(Value),
    Conflict,
    Invalid,
    Forbidden,
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 100 && !id.chars().any(char::is_control)
}

fn load(conn: &Connection, id: &str) -> Result<Option<CaseRecord>> {
    let mut stmt=conn.prepare("SELECT id,action,chat_id,target_user_id,target_name,actor_user_id,actor_name,source_message_id,evidence_text,model_score,matched_rule_id,matched_rule_pattern,status,log_message_id,created_at FROM cases WHERE id=?1")?;
    let mut rows = stmt.query([id])?;
    rows.next()?.map(case_from_row).transpose()
}

fn reversible(case: &CaseRecord) -> bool {
    matches!(
        case.action,
        ActionKind::AutoBan
            | ActionKind::SpamBan
            | ActionKind::ReportApproved
            | ActionKind::GuestBotBan
            | ActionKind::GuestInvokerBan
            | ActionKind::ProjectBan
    ) && !matches!(case.status.as_str(), "reversal_pending" | "reversed")
}

fn review_location(conn: &Connection, id: &str, kind: &str) -> Result<Option<(i64, i32)>> {
    let saved=conn.query_row("SELECT chat_id,message_id FROM review_updates WHERE case_id=?1 AND chat_id IS NOT NULL AND message_id IS NOT NULL",[id],|r|Ok((r.get(0)?,r.get(1)?))).optional()?;
    if saved.is_some() {
        return Ok(saved);
    }
    let sql = if kind == "report" {
        "SELECT review_chat_id,review_message_id FROM report_deliveries WHERE case_id=?1 AND review_message_id IS NOT NULL"
    } else {
        "SELECT chat_id,message_id FROM training_review_locations WHERE case_id=?1 AND chat_id IS NOT NULL AND message_id IS NOT NULL"
    };
    Ok(conn
        .query_row(sql, [id], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?)
}

fn review_state(
    conn: &Connection,
    case: &CaseRecord,
    samples: i64,
    eligible: bool,
) -> Result<Value> {
    let training: Option<(String, i64)> = conn
        .query_row(
            "SELECT decision,actor_id FROM training_reviews WHERE case_id=?1",
            [&case.id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    let mode: Option<String> = conn
        .query_row(
            "SELECT training_mode FROM ban_followups WHERE case_id=?1",
            [&case.id],
            |r| r.get(0),
        )
        .optional()?;
    let is_report = matches!(
        case.action,
        ActionKind::PendingReport | ActionKind::ReportApproved | ActionKind::ReportRejected
    );
    let is_training = case.action == ActionKind::SpamBan
        && (training.is_some()
            || mode.as_deref() == Some("review")
            || (mode.is_none() && samples == 0 && !eligible));
    let kind = if is_report {
        Some("report")
    } else if is_training {
        Some("training")
    } else {
        None
    };
    let decision = if is_report {
        match case.action {
            ActionKind::ReportApproved => Some("approve"),
            ActionKind::ReportRejected => Some("reject"),
            _ => None,
        }
    } else {
        training.as_ref().map(|(decision, _)| decision.as_str())
    };
    let can_review = decision.is_none()
        && if is_report {
            case.action == ActionKind::PendingReport && case.status == "pending_review"
        } else {
            is_training
                && case.matched_rule_pattern.as_deref() != Some("BOTSPAM")
                && !is_empty_ml_text(&case.evidence_text)
                && !matches!(
                    case.status.as_str(),
                    "ban_pending" | "ban_failed" | "reversed" | "reversal_pending"
                )
        };
    let reporter = if is_report && decision.is_none() {
        case.actor_user_id
    } else {
        None
    };
    let (exempt, strikes) = review_decisions::reporter_state(conn, reporter)?;
    let actor = if is_report && decision.is_some() {
        case.actor_user_id
    } else {
        training.as_ref().map(|(_, actor)| *actor)
    };
    let known = if let Some(kind) = kind {
        review_location(conn, &case.id, kind)?.is_some()
    } else {
        false
    };
    Ok(
        json!({"kind":kind,"decision":decision,"actor_id":actor,"can_review":can_review,
        "reporter_id":reporter,"reporter_exempt":exempt,"reporter_strikes":strikes,"strike_limit":REPORT_STRIKE_LIMIT,"message_known":known}),
    )
}

fn details(conn: &Connection, case: &CaseRecord, offset: u32) -> Result<Value> {
    let can_reverse = reversible(case);
    let samples: i64 = conn.query_row(
        "SELECT COUNT(*) FROM training_samples WHERE case_id=?1",
        [&case.id],
        |r| r.get(0),
    )?;
    let eligible: bool = conn.query_row(
        "SELECT netban_eligible FROM cases WHERE id=?1",
        [&case.id],
        |r| r.get(0),
    )?;
    let mut hash = Sha256::new();
    let review = review_state(conn, case, samples, eligible)?;
    hash.update(serde_json::to_vec(case)?);
    hash.update(serde_json::to_vec(&(samples, eligible))?);
    hash.update(serde_json::to_vec(&review)?);
    // Stream the complete impact through the hash while returning only one page.
    let mut stmt = conn.prepare("WITH chats AS (
        SELECT ?2 AS chat_id UNION SELECT chat_id FROM network_ban_targets WHERE case_id=?1
        UNION SELECT chat_id FROM network_deliveries WHERE case_id=?1)
        SELECT chats.chat_id,g.title,d.state,d.attempts,d.last_error,COALESCE(d.outcome_unknown,0),
        EXISTS(SELECT 1 FROM network_ban_targets n WHERE n.case_id=?1 AND n.chat_id=chats.chat_id)
        FROM chats LEFT JOIN group_module_settings g ON g.chat_id=chats.chat_id
        LEFT JOIN network_deliveries d ON d.case_id=?1 AND d.chat_id=chats.chat_id ORDER BY chats.chat_id")?;
    let mut rows = stmt.query(params![case.id, case.chat_id])?;
    let mut targets = Vec::new();
    let (mut total, mut unban, mut retain, mut cancel) = (0_u32, 0_u32, 0_u32, 0_u32);
    while let Some(row) = rows.next()? {
        let chat: i64 = row.get(0)?;
        let state: Option<String> = row.get(2)?;
        let unknown: bool = row.get(5)?;
        let recorded: bool = row.get(6)?;
        let affected = recorded || unknown || (chat == case.chat_id && can_reverse);
        let other: bool = conn.query_row(
            reliability::OTHER_BAN_IN_CHAT,
            params![case.id, chat, case.target_user_id],
            |r| r.get(0),
        )?;
        let action = if affected {
            if other {
                retain += 1;
                "retain"
            } else {
                unban += 1;
                "unban"
            }
        } else if state.as_deref() == Some("pending") {
            "cancel"
        } else {
            "none"
        };
        if state.as_deref() == Some("pending") {
            cancel += 1;
        }
        hash.update(serde_json::to_vec(&(
            chat, &state, unknown, recorded, other,
        ))?);
        if total >= offset && targets.len() < 25 {
            targets.push(json!({"chat_id":chat,"title":row.get::<_,Option<String>>(1)?,
                "state":state,"attempts":row.get::<_,Option<i64>>(3)?,"last_error":row.get::<_,Option<String>>(4)?,
                "outcome_unknown":unknown,"origin":chat==case.chat_id,"reversal_action":action}));
        }
        total += 1;
    }
    // A replaced sample can have the same count but must invalidate confirmation.
    let mut sample_stmt =
        conn.prepare("SELECT id,label,text FROM training_samples WHERE case_id=?1 ORDER BY id")?;
    let mut sample_rows = sample_stmt.query([&case.id])?;
    while let Some(row) = sample_rows.next()? {
        hash.update(serde_json::to_vec(&(
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))?);
    }
    Ok(
        json!({"case":{"id":case.id,"action":case.action.as_str(),"chat_id":case.chat_id,
        "target_user_id":case.target_user_id,"target_name":case.target_name,"status":case.status,
        "evidence":case.evidence_text,"reason":case.matched_rule_pattern,"model_score":case.model_score,
        "netban_eligible":eligible,"created_at":case.created_at},
        "revision":format!("{:x}",hash.finalize()),"can_reverse":can_reverse,
        "review":review,
        "training_samples":samples,"unban_count":unban,"retained_count":retain,"cancel_count":cancel,
        "targets":targets,"total":total,"offset":offset,"has_more":u64::from(offset)+25<u64::from(total)}),
    )
}

fn redact(value: &mut Value, config: &Config) {
    match value {
        Value::String(s) => {
            let mut clean = s.clone();
            for secret in [
                Some(config.bot_token.as_str()),
                config.hostctl_secret.as_deref(),
            ]
            .into_iter()
            .flatten()
            .filter(|s| !s.is_empty())
            {
                clean = clean.replace(secret, "[redacted]");
            }
            *s = notices::preview(&clean, 8192);
        }
        Value::Array(items) => items.iter_mut().for_each(|item| redact(item, config)),
        Value::Object(fields) => fields.values_mut().for_each(|item| redact(item, config)),
        _ => {}
    }
}

impl Runtime {
    pub(super) fn migrate_v30_to_v31(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS host_case_requests (
            request_id TEXT PRIMARY KEY,actor_id INTEGER NOT NULL,payload TEXT NOT NULL,
            result TEXT NOT NULL,action_id INTEGER NOT NULL);
            PRAGMA user_version=31;",
        )?;
        tx.commit()?;
        Ok(())
    }

    pub(super) async fn host_case(&self, query: Query) -> Result<Option<Value>> {
        anyhow::ensure!(query.valid(), "invalid case query");
        let mut config = self.config.clone();
        config.hostctl_secret = self.hostctl_secret().await;
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            let Some(case) = load(&tx, &query.case_id)? else {
                return Ok(None);
            };
            let mut data = details(&tx, &case, query.offset)?;
            redact(&mut data, &config);
            Ok(Some(data))
        })
        .await
    }

    pub(super) async fn reverse_host_case(&self, actor: i64, request: Reverse) -> Result<Outcome> {
        if !is_host(actor) {
            return Ok(Outcome::Forbidden);
        }
        if !valid_id(&request.case_id)
            || !role_updates::valid_user(request.target_user_id)
            || Uuid::parse_str(&request.request_id).is_err()
            || request.expected_revision.len() != 64
            || !request
                .expected_revision
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
        {
            return Ok(Outcome::Invalid);
        }
        let review = self.review_guard(&request.case_id).await;
        let user = self.user_action_guard(request.target_user_id).await;
        self.with_conn(move |conn| {
            // These guards stay owned by the transaction if the HTTP request disappears.
            let _guards=(review,user);
            let tx=conn.transaction()?;
            let payload=serde_json::to_string(&request)?;
            let receipt: Option<(i64,String,String)>=tx.query_row("SELECT actor_id,payload,result FROM host_case_requests WHERE request_id=?1",[&request.request_id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
            if let Some((who,original,result))=receipt {
                return Ok(if who==actor && original==payload {Outcome::Saved(serde_json::from_str(&result)?)} else {Outcome::Conflict});
            }
            let Some(case)=load(&tx,&request.case_id)? else {return Ok(Outcome::Invalid)};
            if case.target_user_id!=request.target_user_id {return Ok(Outcome::Conflict)}
            let current=details(&tx,&case,0)?;
            if current["revision"]!=request.expected_revision {return Ok(Outcome::Conflict)}
            if !reversible(&case) {return Ok(Outcome::Invalid)}
            reversal_retry::begin_reversal_tx(&tx,&case,actor,"項目主持人")?;
            let result=json!({"case_id":case.id,"status":"reversal_pending","request_id":request.request_id});
            let summary=format!("撤銷案件 {}；預計解封 {} 群、保留 {} 群、取消 {} 項聯防、移除 {} 筆樣本",case.id,current["unban_count"],current["retained_count"],current["cancel_count"],current["training_samples"]);
            tx.execute("INSERT INTO maintainer_actions(actor_id,actor_name,chat_id,command,summary,undo_data,created_at) VALUES (?1,'項目主持人',?2,'面板撤銷案件',?3,?4,?5)",params![actor,case.chat_id,summary,serde_json::to_string(&UndoData::NotRevertible)?,Utc::now().to_rfc3339()])?;
            tx.execute("INSERT INTO host_case_requests(request_id,actor_id,payload,result,action_id) VALUES (?1,?2,?3,?4,?5)",params![request.request_id,actor,payload,serde_json::to_string(&result)?,tx.last_insert_rowid()])?;
            tx.commit()?;
            Ok(Outcome::Saved(result))
        }).await
    }

    pub(super) async fn review_host_case(&self, actor: i64, request: Review) -> Result<Outcome> {
        if !is_host(actor) {
            return Ok(Outcome::Forbidden);
        }
        if !valid_id(&request.case_id)
            || !role_updates::valid_user(request.target_user_id)
            || Uuid::parse_str(&request.request_id).is_err()
            || request.expected_revision.len() != 64
            || !request
                .expected_revision
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
            || !matches!(request.kind.as_str(), "report" | "training")
            || !matches!(request.decision.as_str(), "approve" | "reject")
        {
            return Ok(Outcome::Invalid);
        }
        let review = self.review_guard(&request.case_id).await;
        let user = self.user_action_guard(request.target_user_id).await;
        let test_group = self.config.test_group_id;
        self.with_model_transaction(move |tx| {
            let guards=(review,user);
            let payload=serde_json::to_string(&request)?;
            let receipt:Option<(i64,String,String)>=tx.query_row("SELECT actor_id,payload,result FROM host_case_requests WHERE request_id=?1",[&request.request_id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
            if let Some((who,original,result))=receipt {
                return Ok((if who==actor && original==payload {Outcome::Saved(serde_json::from_str(&result)?)} else {Outcome::Conflict},guards));
            }
            let Some(case)=load(tx,&request.case_id)? else {return Ok((Outcome::Invalid,guards));};
            if case.target_user_id!=request.target_user_id {return Ok((Outcome::Conflict,guards));}
            let current=details(tx,&case,0)?;
            if current["revision"]!=request.expected_revision {return Ok((Outcome::Conflict,guards));}
            if current["review"]["can_review"]!=true || current["review"]["kind"]!=request.kind {return Ok((Outcome::Invalid,guards));}
            let location=review_location(tx,&case.id,&request.kind)?;
            let changed=if request.kind=="report" {
                review_decisions::report(tx,&case.id,&request.decision,(actor,"項目主持人"),location)?
            } else {review_decisions::training(tx,&case.id,&request.decision,actor,location,test_group)?};
            anyhow::ensure!(changed,"review decision changed within transaction");
            let status:String=tx.query_row("SELECT status FROM cases WHERE id=?1",[&case.id],|r|r.get(0))?;
            let result=json!({"case_id":case.id,"kind":request.kind,"decision":request.decision,"status":status,"request_id":request.request_id});
            let summary=format!("案件 {}；{}：{}",case.id,if request.kind=="report" {"舉報"} else {"訓練"},if request.decision=="approve" {"批准"} else {"拒絕"});
            tx.execute("INSERT INTO maintainer_actions(actor_id,actor_name,chat_id,command,summary,undo_data,created_at) VALUES (?1,'項目主持人',?2,'面板審核',?3,?4,?5)",params![actor,case.chat_id,summary,serde_json::to_string(&UndoData::NotRevertible)?,Utc::now().to_rfc3339()])?;
            tx.execute("INSERT INTO host_case_requests(request_id,actor_id,payload,result,action_id) VALUES (?1,?2,?3,?4,?5)",params![request.request_id,actor,payload,serde_json::to_string(&result)?,tx.last_insert_rowid()])?;
            Ok((Outcome::Saved(result),guards))
        }).await.map(|(outcome,_)|outcome)
    }
}
