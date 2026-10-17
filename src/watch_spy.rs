use super::*;
use rusqlite::OptionalExtension;

pub(super) const MARKER: &str = "WATCH_SPY";

pub(super) fn is_report_chat(chat: i64) -> bool {
    env::var("WATCH_REVIEW_CHAT_ID")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        == Some(chat)
}

pub(super) fn report_query(conn: &Connection, message: &Message) -> Result<Option<String>> {
    let Some(user) = message.from.as_ref() else {
        return Ok(None);
    };
    if user.is_bot
        || message.sender_chat.is_some()
        || message.via_bot.is_some()
        || message.forward_origin().is_some()
        || message.edit_date().is_some()
    {
        return Ok(None);
    }
    let text = message.text().unwrap_or("");
    let first = text.split_whitespace().next().unwrap_or("");
    if let Some((_, username)) = first.split_once('@') {
        if !username.eq_ignore_ascii_case("WikiBlacklistRobot") {
            return Ok(None);
        }
    }
    let ModerationCommand::CaseLookup(reference) = parse_command(text) else {
        return Ok(None);
    };
    if !authorized(conn, user.id.0 as i64)? || destination(conn)? != message.chat.id.0 {
        return Ok(None);
    }
    Ok(Some(reference))
}

pub(super) async fn handle_report_message(
    bot: &Bot,
    runtime: &Arc<Runtime>,
    message: &Message,
) -> ResponseResult<()> {
    let reference =
        (|| -> Result<Option<String>> { report_query(&watch_inbox::private_db()?, message) })()
            .ok()
            .flatten();
    let Some(reference) = reference else {
        return Ok(());
    };
    let text = if reference.is_empty() {
        "用法：<code>/case #案件碼</code>".to_string()
    } else {
        match runtime.load_case_reference(&reference).await {
            Ok(Some(case)) => {
                let link = case
                    .log_message_id
                    .filter(|id| *id > 0)
                    .map(|id| public_log_link(&runtime.config, id))
                    .unwrap_or_else(|| "-".to_string());
                let reason = runtime
                    .blacklist_reason_link()
                    .await
                    .unwrap_or_else(|| link.clone());
                format_case_lookup(&case, &link, &reason)
            }
            Err(err) if err.downcast_ref::<case_ids::Ambiguous>().is_some() => {
                case_ids::AMBIGUOUS.to_string()
            }
            Ok(None) => "找不到這個案件。".to_string(),
            Err(_) => "暫時無法讀取案件，請稍後再試。".to_string(),
        }
    };
    bot.send_message(message.chat.id, text)
        .parse_mode(ParseMode::Html)
        .await?;
    Ok(())
}

pub(super) fn is_case(case: &CaseRecord) -> bool {
    case.matched_rule_pattern.as_deref() == Some(MARKER)
}

pub(super) fn settings(conn: &Connection) -> Result<serde_json::Value> {
    let value: String =
        conn.query_row("SELECT value FROM meta WHERE key='spy_settings'", [], |r| {
            r.get(0)
        })?;
    Ok(serde_json::from_str(&value)?)
}

pub(super) fn authorized(conn: &Connection, actor: i64) -> Result<bool> {
    let owner: String =
        conn.query_row("SELECT value FROM meta WHERE key='watch_owner'", [], |r| {
            r.get(0)
        })?;
    Ok(serde_json::from_str::<i64>(&owner)? == actor
        || conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sudo_users WHERE user_id=?1)",
            [actor],
            |r| r.get::<_, bool>(0),
        )?)
}

fn destination(conn: &Connection) -> Result<i64> {
    let verified: String = conn.query_row(
        "SELECT value FROM meta WHERE key='watch_group_verified_at'",
        [],
        |r| r.get(0),
    )?;
    let age = Utc::now().timestamp() - serde_json::from_str::<i64>(&verified)?;
    anyhow::ensure!(
        (0..=90).contains(&age),
        "WATCH report group verification expired"
    );
    settings(conn)?["destination"]
        .as_i64()
        .filter(|v| *v < 0)
        .context("missing WATCH report group")
}

pub(super) fn allowed_location(actor: i64, chat: i64) -> bool {
    (|| -> Result<bool> {
        let conn = watch_inbox::private_db()?;
        Ok(authorized(&conn, actor)? && destination(&conn)? == chat)
    })()
    .unwrap_or(false)
}

pub(super) fn review(
    tx: &rusqlite::Transaction<'_>,
    id: &str,
    decision: &str,
    actor: i64,
    location: Option<(i64, i32)>,
) -> Result<bool> {
    let spy: bool = tx
        .query_row(
            "SELECT matched_rule_pattern='WATCH_SPY' FROM cases WHERE id=?1",
            [id],
            |r| r.get::<_, Option<bool>>(0),
        )?
        .unwrap_or(false);
    if !spy {
        return Ok(false);
    }
    let bridge = watch_inbox::private_db()?;
    // Serialize the authority check with /sudo, without holding this lock while
    // waiting for the main database or for Telegram.
    bridge.execute_batch("BEGIN IMMEDIATE")?;
    review_with_bridge(tx, &bridge, id, decision, actor, location)?;
    bridge.execute_batch("COMMIT")?;
    Ok(true)
}

pub(super) fn review_with_bridge(
    tx: &rusqlite::Transaction<'_>,
    bridge: &Connection,
    id: &str,
    decision: &str,
    actor: i64,
    location: Option<(i64, i32)>,
) -> Result<()> {
    anyhow::ensure!(
        matches!(decision, "approve" | "reject"),
        "invalid WATCH decision"
    );
    anyhow::ensure!(
        authorized(bridge, actor)?,
        "只限主持人及 WATCH 授權人員審核"
    );
    let (chat, message) = location.context("WATCH review needs its original card")?;
    anyhow::ensure!(destination(bridge)? == chat, "wrong WATCH review group");
    let valid: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM report_deliveries WHERE case_id=?1 AND review_chat_id=?2 AND review_message_id=?3)", params![id,chat,message], |r| r.get(0))?;
    anyhow::ensure!(valid, "wrong WATCH review card");
    let user: i64 = tx.query_row("SELECT target_user_id FROM cases WHERE id=?1 AND matched_rule_pattern='WATCH_SPY' AND action='pending_report' AND status='pending_review'", [id], |r| r.get(0))?;
    if decision == "approve" {
        anyhow::ensure!(
            !watch_inbox::protected_in_spb(tx, user)?
                && !watch_inbox::protected_in_bridge(bridge, user)?,
            "此帳號目前受保護"
        );
        tx.execute("UPDATE cases SET action='report_approved',status='approved_and_banned',actor_user_id=?2,actor_name='WATCH 審核人員' WHERE id=?1",params![id,actor])?;
        network_delivery::enqueue_network_ban(tx, id, None)?;
    } else {
        tx.execute("UPDATE cases SET action='report_rejected',status='rejected_and_cleaned',actor_user_id=?2,actor_name='WATCH 審核人員' WHERE id=?1",params![id,actor])?;
    }
    Ok(())
}

pub(super) fn card(case: &CaseRecord, chat: i64) -> Result<String> {
    review_card(
        case,
        chat,
        "WATCH · 待審核訊息",
        "核准：加入 SPB 及 hexDrill +spb。拒絕：不封禁。來源群組不執行任何動作；不訓練模型。",
        None,
    )
}

pub(super) fn review_card(
    case: &CaseRecord,
    chat: i64,
    title: &str,
    note: &str,
    reviewer: Option<i64>,
) -> Result<String> {
    let bridge = watch_inbox::private_db()?;
    anyhow::ensure!(
        destination(&bridge)? == chat,
        "WATCH report group not verified"
    );
    let text: String = bridge.query_row(
        "SELECT text FROM watch_spy WHERE id=?1 AND chat_id=?2 AND user_id=?3",
        params![case.id, case.chat_id, case.target_user_id],
        |r| r.get(0),
    )?;
    let mut display = case.clone();
    display.evidence_text = text;
    let mut body = notices::review_card(&display, title, note, reviewer);
    body = body.replace(
        "（證據過長，完整內容已保存在案例記錄。）",
        "（內容過長，請開啟來源訊息查看全文。）",
    );
    if let Some(message) = case.source_message_id {
        if let Some(group) = case.chat_id.to_string().strip_prefix("-100") {
            body.push_str(&format!(
                "\n<a href=\"https://t.me/c/{group}/{message}\">來源訊息</a>"
            ));
        }
    }
    Ok(body)
}

pub(super) struct Incoming {
    pub id: String,
    pub chat: i64,
    pub message: i32,
    pub user: i64,
    pub text: String,
}

pub(super) async fn process(
    runtime: &Runtime,
    item: &Incoming,
    review_chat: i64,
    protected: bool,
) -> Result<(String, f64, f64)> {
    anyhow::ensure!(
        Uuid::parse_str(&item.id).is_ok()
            && item.chat < 0
            && item.chat != review_chat
            && item.message > 0
            && item.user > 0
            && item.user < (1_i64 << 52)
            && item.text.chars().count() <= 4096,
        "invalid WATCH message"
    );
    let threshold = runtime.effective_threshold(None).await?;
    anyhow::ensure!(
        threshold.is_finite() && (0.5..=1.0).contains(&threshold),
        "invalid WATCH global threshold"
    );
    let score = {
        let model = runtime.model.lock().await;
        score_spam_from_text(&model, &item.text)
    };
    anyhow::ensure!(
        score.is_finite() && (0.0..=1.0).contains(&score),
        "invalid WATCH score"
    );
    let (id, chat, message, user) = (item.id.clone(), item.chat, item.message, item.user);
    let guard = runtime.user_action_guard(user).await;
    let status = runtime.with_conn(move |conn| {
        let _guard = guard;
        let tx = conn.transaction()?;
        let existing: Option<String> = tx.query_row("SELECT status FROM cases WHERE id=?1",[&id],|r|r.get(0)).optional()?;
        if let Some(status) = existing { return Ok(if status == "pending_review" {"review"} else {"done"}.to_string()); }
        if protected || watch_inbox::protected_in_spb(&tx,user)? {return Ok("exempt".into());}
        if !passes_threshold(score,threshold) {return Ok("ham".into());}
        anyhow::ensure!(!operations::controls(&tx)?.automatic_new_paused, "WATCH intake paused");
        let repeated: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM cases WHERE target_user_id=?1 AND matched_rule_pattern='WATCH_SPY' AND (status='pending_review' OR (action='report_approved' AND status='approved_and_banned')))",[user],|r|r.get(0))?;
        if repeated {return Ok("exempt".into());}
        tx.execute("INSERT INTO cases(id,action,chat_id,target_user_id,target_name,source_message_id,evidence_text,model_score,matched_rule_pattern,status,created_at)
            VALUES (?1,'pending_report',?2,?3,?4,?5,'WATCH 群組掃描；原文只在指定私密群組供審核。',?6,'WATCH_SPY','pending_review',?7)",
            params![id,chat,user,user.to_string(),message,score,Utc::now().to_rfc3339()])?;
        case_thresholds::record(&tx,&id,"detection",threshold,score)?;
        tx.execute("INSERT INTO report_deliveries(case_id,command_id,review_chat_id) VALUES (?1,0,?2)",params![id,review_chat])?;
        tx.commit()?; Ok("review".into())
    }).await?;
    Ok((status, score, threshold))
}

async fn tick(runtime: &Runtime) -> Result<()> {
    let bridge = watch_inbox::private_db()?;
    let settings = settings(&bridge)?;
    let review_chat = settings["destination"]
        .as_i64()
        .context("missing destination")?;
    let items = {
        let mut stmt = bridge.prepare("SELECT id,chat_id,message_id,user_id,text FROM watch_spy WHERE status='pending' ORDER BY received_at,id LIMIT 50")?;
        let rows = stmt
            .query_map([], |r| {
                Ok(Incoming {
                    id: r.get(0)?,
                    chat: r.get(1)?,
                    message: r.get(2)?,
                    user: r.get(3)?,
                    text: r.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        rows
    };
    for item in items {
        let ignored = item.chat == review_chat
            || settings["excluded"]
                .as_array()
                .context("missing exclusions")?
                .iter()
                .any(|v| v.as_i64() == Some(item.chat))
            || !bridge.query_row(
                "SELECT EXISTS(SELECT 1 FROM spy_groups WHERE chat_id=?1)",
                [item.chat],
                |r| r.get::<_, bool>(0),
            )?
            || bridge.query_row(
                "SELECT EXISTS(SELECT 1 FROM spy_ignored WHERE chat_id=?1)",
                [item.chat],
                |r| r.get::<_, bool>(0),
            )?;
        let protected = ignored || watch_inbox::protected_in_bridge(&bridge, item.user)?;
        let (state, score, threshold) = process(runtime, &item, review_chat, protected).await?;
        bridge.execute("UPDATE watch_spy SET status=?2,score=?3,threshold=?4,text=CASE WHEN ?2 IN ('ham','exempt') THEN '' ELSE text END WHERE id=?1 AND status='pending'",params![item.id,state,score,threshold])?;
    }
    Ok(())
}

pub(super) fn spawn(runtime: Arc<Runtime>) {
    if env::var("WATCH_SPY_ENABLED").as_deref() != Ok("1") {
        return;
    }
    tokio::spawn(async move {
        loop {
            if let Err(error) = tick(&runtime).await {
                log::warn!(
                    "WATCH scan: {}",
                    notices::diagnostic(&runtime.config, &error.to_string())
                );
            }
            sleep(Duration::from_secs(3)).await;
        }
    });
}
