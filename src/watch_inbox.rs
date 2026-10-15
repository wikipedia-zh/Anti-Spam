use super::*;
use rusqlite::{OpenFlags, OptionalExtension};

pub(super) const MARKER: &str = "WATCH_DM";

pub(super) fn is_case(case: &CaseRecord) -> bool {
    case.chat_id == case.target_user_id
        && case.chat_id > 0
        && case.matched_rule_pattern.as_deref() == Some(MARKER)
}

pub(super) fn private_db() -> Result<Connection> {
    let conn = Connection::open_with_flags(
        env::var("HEXDRILL_STATE_DB")?,
        OpenFlags::SQLITE_OPEN_READ_WRITE,
    )?;
    conn.busy_timeout(Duration::from_secs(2))?;
    conn.execute_batch("PRAGMA trusted_schema=OFF;")?;
    Ok(conn)
}

pub(super) fn protected_in_bridge(conn: &Connection, user: i64) -> Result<bool> {
    let binding: String =
        conn.query_row("SELECT value FROM meta WHERE key='binding'", [], |row| {
            row.get(0)
        })?;
    let binding: serde_json::Value = serde_json::from_str(&binding)?;
    let owner: String = conn.query_row(
        "SELECT value FROM meta WHERE key='watch_owner'",
        [],
        |row| row.get(0),
    )?;
    let owner: i64 = serde_json::from_str(&owner)?;
    Ok(user == owner
        || binding["account_id"].as_i64() == Some(user)
        || binding["hex_bot_id"].as_i64() == Some(user)
        || conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sudo_users WHERE user_id=?1)",
            [user],
            |row| row.get::<_, bool>(0),
        )?)
}

pub(super) fn protected_in_spb(conn: &Connection, user: i64) -> Result<bool> {
    Ok(is_host(user) || is_platform_pseudo_user(user) || conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM global_whitelist WHERE user_id=?1) OR EXISTS(SELECT 1 FROM maintainers WHERE user_id=?1) OR EXISTS(SELECT 1 FROM reviewers WHERE user_id=?1)", [user], |row| row.get::<_,bool>(0))?)
}

pub(super) fn review(
    tx: &rusqlite::Transaction<'_>,
    id: &str,
    decision: &str,
    actor: i64,
) -> Result<bool> {
    let user: Option<i64> = tx.query_row("SELECT target_user_id FROM cases WHERE id=?1 AND matched_rule_pattern='WATCH_DM' AND chat_id=target_user_id AND chat_id>0", [id], |row| row.get(0)).optional()?;
    let Some(user) = user else {
        return Ok(false);
    };
    anyhow::ensure!(is_host(actor), "私訊案件只限主持人審核");
    if decision == "approve" {
        approve(tx, &private_db()?, id, user, actor)?;
    } else {
        tx.execute("UPDATE cases SET action='report_rejected',status='rejected_and_cleaned',actor_user_id=?2,actor_name='項目主持人' WHERE id=?1", params![id,actor])?;
    }
    Ok(true)
}

pub(super) fn approve(
    tx: &rusqlite::Transaction<'_>,
    bridge: &Connection,
    id: &str,
    user: i64,
    actor: i64,
) -> Result<()> {
    anyhow::ensure!(is_host(actor), "私訊案件只限主持人審核");
    anyhow::ensure!(
        !protected_in_spb(tx, user)? && !protected_in_bridge(bridge, user)?,
        "此帳號目前受保護，不能加入聯防黑名單"
    );
    tx.execute("UPDATE cases SET action='report_approved',status='approved_and_banned',actor_user_id=?2,actor_name='項目主持人' WHERE id=?1",params![id,actor])?;
    network_delivery::enqueue_network_ban(tx, id, None)?;
    Ok(())
}

pub(super) struct Incoming {
    pub id: String,
    pub user: i64,
    pub text: String,
}

pub(super) async fn process(
    runtime: &Runtime,
    incoming: &Incoming,
    protected: bool,
) -> Result<(String, f64, f64)> {
    anyhow::ensure!(
        Uuid::parse_str(&incoming.id).is_ok()
            && incoming.user > 0
            && incoming.user < (1_i64 << 52)
            && incoming.text.chars().count() <= 4096,
        "invalid private message record"
    );
    let threshold = runtime.effective_threshold(None).await?;
    anyhow::ensure!(
        threshold.is_finite() && (0.5..=1.0).contains(&threshold),
        "private message automation requires a high global threshold"
    );
    let score = {
        let model = runtime.model.lock().await;
        score_spam_from_text(&model, &incoming.text)
    };
    anyhow::ensure!(
        score.is_finite() && (0.0..=1.0).contains(&score),
        "invalid private message score"
    );
    let regex_hit = runtime
        .spam_rules
        .read()
        .await
        .iter()
        .any(|rule| regex_is_match(&rule.regex, &incoming.text));
    let id = incoming.id.clone();
    let user = incoming.user;
    let test_group = runtime.config.test_group_id;
    let review_chat = runtime.config.report_channel_id;
    let _guard = runtime.user_action_guard(user).await;
    let outcome = runtime.with_conn(move |conn| {
        let tx = conn.transaction()?;
        // An existing case, including a reversed case, is never recreated on replay.
        let existing: Option<String> = tx.query_row("SELECT status FROM cases WHERE id=?1", [&id], |row| row.get(0)).optional()?;
        if let Some(status) = existing {
            return Ok(if status == "pending_review" {"review"} else if matches!(status.as_str(),"auto_banned"|"approved_and_banned") {"banned"} else {"exempt"}.to_string());
        }
        if protected || protected_in_spb(&tx,user)? { return Ok("exempt".to_string()); }
        if score < 0.5 && !regex_hit { return Ok("ham".to_string()); }
        anyhow::ensure!(!operations::controls(&tx)?.automatic_new_paused, "automatic admission is paused");
        let recent: i64 = tx.query_row("SELECT count(*) FROM cases WHERE matched_rule_pattern='WATCH_DM' AND created_at>?1", [ (Utc::now()-chrono::Duration::days(1)).to_rfc3339() ], |row| row.get(0))?;
        let automatic = passes_threshold(score,threshold) && recent < 20;
        let action = if automatic {"auto_ban"} else {"pending_report"};
        let status = if automatic {"auto_banned"} else {"pending_review"};
        let evidence = format!("Spam Protection Bot 角色帳號收到的私訊。原文僅主持人可用 /evidence {} 查閱。", case_ids::display(&id));
        tx.execute("INSERT INTO cases(id,action,chat_id,target_user_id,target_name,evidence_text,model_score,matched_rule_pattern,status,created_at)
            VALUES (?1,?2,?3,?3,?4,?5,?6,'WATCH_DM',?7,?8)",params![id,action,user,user.to_string(),evidence,score,status,Utc::now().to_rfc3339()])?;
        case_thresholds::record(&tx,&id,"detection",threshold,score)?;
        if automatic {
            case_thresholds::record(&tx,&id,"network",threshold,score)?;
            network_delivery::enqueue_network_ban(&tx,&id,test_group)?;
        } else {
            tx.execute("INSERT INTO report_deliveries(case_id,command_id,review_chat_id) VALUES (?1,0,?2)",params![id,review_chat])?;
        }
        tx.commit()?;
        Ok(if automatic {"banned"} else {"review"}.to_string())
    }).await?;
    Ok((outcome, score, threshold))
}

async fn tick(runtime: &Runtime) -> Result<()> {
    let conn = private_db()?;
    let incoming = {
        let mut stmt = conn.prepare("SELECT id,user_id,text FROM watch_inbox WHERE status='pending' ORDER BY received_at,id LIMIT 10")?;
        let rows = stmt
            .query_map([], |r| {
                Ok(Incoming {
                    id: r.get(0)?,
                    user: r.get(1)?,
                    text: r.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        rows
    };
    for item in incoming {
        // Serialize admission with /sudo changes in the private ledger.
        conn.execute_batch("BEGIN IMMEDIATE")?;
        let protected = protected_in_bridge(&conn, item.user)?;
        let result = process(runtime, &item, protected).await;
        let (status, score, threshold) = match result {
            Ok(value) => value,
            Err(error) => {
                conn.execute_batch("ROLLBACK")?;
                return Err(error);
            }
        };
        conn.execute("UPDATE watch_inbox SET status=?2,score=?3,threshold=?4,processed_at=?5 WHERE id=?1 AND status='pending'", params![item.id,status,score,threshold,Utc::now().timestamp()])?;
        conn.execute_batch("COMMIT")?;
    }
    Ok(())
}

pub(super) fn spawn(runtime: Arc<Runtime>) {
    if env::var("WATCH_DM_ENABLED").as_deref() != Ok("1") {
        return;
    }
    tokio::spawn(async move {
        loop {
            if let Err(error) = tick(&runtime).await {
                log::warn!(
                    "watch inbox: {}",
                    notices::diagnostic(&runtime.config, &error.to_string())
                );
            }
            sleep(Duration::from_secs(3)).await;
        }
    });
}
