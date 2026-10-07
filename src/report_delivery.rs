use super::*;
use rusqlite::OptionalExtension;

pub(super) enum Submission {
    Queued(String),
    Suspended(i64),
}

impl Runtime {
    pub(super) fn migrate_v25_to_v26(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS report_deliveries (
                case_id TEXT PRIMARY KEY,command_id INTEGER NOT NULL,review_chat_id INTEGER NOT NULL,
                review_message_id INTEGER,confirmation_message_id INTEGER,
                state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','done')),
                attempts INTEGER NOT NULL DEFAULT 0,next_attempt_at INTEGER NOT NULL DEFAULT 0,last_error TEXT
             );
             CREATE INDEX IF NOT EXISTS idx_report_delivery_due ON report_deliveries(state,next_attempt_at);
             PRAGMA user_version=26;",
        )?;
        tx.commit()?;
        Ok(())
    }

    pub(super) async fn queue_report(
        &self,
        case: CaseRecord,
        command_id: i32,
    ) -> Result<Submission> {
        anyhow::ensure!(
            case.action == ActionKind::PendingReport,
            "invalid report action"
        );
        let actor = case.actor_user_id.context("missing reporter")?;
        let exempt = self.is_maintainer(actor).await;
        let review_chat = self.config.report_channel_id;
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            let existing = tx.query_row(
                "SELECT case_id FROM moderation_requests WHERE chat_id=?1 AND message_id=?2",
                params![case.chat_id,command_id], |r| r.get::<_,String>(0),
            ).optional()?;
            if let Some(id) = existing { return Ok(Submission::Queued(id)); }
            let strikes = tx.query_row("SELECT rejected_count FROM report_offenses WHERE user_id=?1",
                [actor], |r| r.get::<_,i64>(0)).optional()?.unwrap_or(0);
            if strikes >= REPORT_STRIKE_LIMIT && !exempt { return Ok(Submission::Suspended(strikes)); }
            tx.execute("INSERT INTO cases(id,action,chat_id,target_user_id,target_name,actor_user_id,actor_name,
                source_message_id,evidence_text,status,created_at) VALUES (?1,'pending_report',?2,?3,?4,?5,?6,?7,?8,'pending_review',?9)",
                params![case.id,case.chat_id,case.target_user_id,case.target_name,actor,case.actor_name,
                    case.source_message_id,case.evidence_text,case.created_at.to_rfc3339()])?;
            tx.execute("INSERT INTO report_deliveries(case_id,command_id,review_chat_id) VALUES (?1,?2,?3)",
                params![case.id,command_id,review_chat])?;
            tx.execute("INSERT INTO moderation_requests(chat_id,message_id,case_id) VALUES (?1,?2,?3)",
                params![case.chat_id,command_id,case.id])?;
            tx.commit()?;
            Ok(Submission::Queued(case.id))
        }).await
    }
}

pub(super) fn remember_confirmation(
    tx: &rusqlite::Transaction<'_>,
    id: &str,
    chat: i64,
    message: i32,
) -> Result<()> {
    let updated = tx.execute(
        "UPDATE review_updates SET confirmation_chat_id=?2,confirmation_message_id=?3,
        confirmation_status='',next_attempt_at=0 WHERE case_id=?1",
        params![id, chat, message],
    )?;
    if updated == 0 {
        tx.execute("INSERT INTO report_confirmations(case_id,chat_id,message_id) VALUES (?1,?2,?3)
            ON CONFLICT(case_id) DO UPDATE SET chat_id=excluded.chat_id,message_id=excluded.message_id",
            params![id,chat,message])?;
    }
    Ok(())
}

pub(super) async fn handle(bot: &Bot, runtime: &Runtime, message: &Message) -> ResponseResult<()> {
    let Some(from) = message.from.as_ref() else {
        return Ok(());
    };
    let Some((target_id, target_name, source_id, evidence_text)) =
        extract_reply_context(message).await
    else {
        reply_ephemeral(bot, message, "請回覆一條疑似 spam 的訊息。").await?;
        return Ok(());
    };
    let case = CaseRecord {
        id: Uuid::new_v4().to_string(),
        action: ActionKind::PendingReport,
        chat_id: message.chat.id.0,
        target_user_id: target_id,
        target_name,
        actor_user_id: Some(from.id.0 as i64),
        actor_name: Some(short_user(from)),
        source_message_id: Some(source_id),
        evidence_text,
        model_score: None,
        matched_rule_id: None,
        matched_rule_pattern: None,
        status: "pending_review".into(),
        log_message_id: None,
        created_at: Utc::now(),
    };
    match runtime.queue_report(case, message.id.0).await {
        Ok(Submission::Queued(id)) => {
            if let Err(err) = deliver(bot, runtime, &id).await {
                log::warn!(
                    "report delivery {id}: {}",
                    notices::diagnostic(&runtime.config, &err.to_string())
                );
            }
            if let Err(err) =
                moderation_queue::deliver_review_updates(bot, runtime, Some(&id)).await
            {
                log::warn!(
                    "report outcome {id}: {}",
                    notices::diagnostic(&runtime.config, &err.to_string())
                );
            }
        }
        Ok(Submission::Suspended(strikes)) => {
            reply_ephemeral(bot,message,format!("你已有 {strikes} 次舉報被拒絕，已暫停使用 /spam。如有疑問請透過 @SEELE_01_BOT 聯絡項目組。")).await?;
            let _ = bot.delete_message(message.chat.id, message.id).await;
        }
        Err(err) => {
            log::warn!(
                "could not save report: {}",
                notices::diagnostic(&runtime.config, &err.to_string())
            );
            reply_ephemeral(bot, message, "舉報未能保存，請稍後重試。").await?;
        }
    }
    Ok(())
}

pub(super) async fn retry(bot: &Bot, runtime: &Runtime) -> Result<()> {
    let ids = runtime.with_conn(|conn| {
        let mut stmt = conn.prepare("SELECT case_id FROM report_deliveries WHERE state='pending' AND next_attempt_at<=?1
            AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?1 ORDER BY next_attempt_at,case_id LIMIT 20")?;
        let rows = stmt.query_map([Utc::now().timestamp()], |r| r.get::<_,String>(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }).await?;
    for id in ids {
        if let Err(err) = deliver(bot, runtime, &id).await {
            log::warn!(
                "report delivery {id}: {}",
                notices::diagnostic(&runtime.config, &err.to_string())
            );
        }
    }
    Ok(())
}

pub(super) async fn deliver(bot: &Bot, runtime: &Runtime, id: &str) -> Result<()> {
    let guard = Arc::new(runtime.review_guard(id).await);
    let case_id = id.to_string();
    let claim_guard = guard.clone();
    let item = runtime.with_conn(move |conn| {
        let _guard = claim_guard;
        let tx = conn.transaction()?;
        let now = Utc::now().timestamp();
        let row = tx.query_row("SELECT command_id,review_chat_id,review_message_id,confirmation_message_id,attempts
            FROM report_deliveries WHERE case_id=?1 AND state='pending' AND next_attempt_at<=?2
            AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?2",params![case_id,now],
            |r| Ok((r.get::<_,i32>(0)?,r.get::<_,i64>(1)?,r.get::<_,Option<i32>>(2)?,r.get::<_,Option<i32>>(3)?,r.get::<_,u32>(4)?))).optional()?;
        if let Some(row) = &row {
            let attempt = row.4.saturating_add(1).min(31);
            let delay = (60_i64 << attempt.saturating_sub(1).min(6)).min(3600);
            tx.execute("UPDATE report_deliveries SET attempts=?2,next_attempt_at=?3 WHERE case_id=?1",params![case_id,attempt,now+delay])?;
        }
        tx.commit()?;
        Ok(row)
    }).await?;
    let Some((command, review_chat, review_message, confirmation, _)) = item else {
        return Ok(());
    };
    let case = runtime
        .load_case(id)
        .await?
        .context("report case missing")?;
    let pending = case.action == ActionKind::PendingReport && case.status == "pending_review";
    // A reviewer may have acted on a card whose send response was lost.
    // The decision already owns that card; never send a fresh pending card.
    if review_message.is_none() && pending {
        let keyboard = InlineKeyboardMarkup::new(vec![vec![
            InlineKeyboardButton::callback("受理並封禁", format!("review:approve:{id}")),
            InlineKeyboardButton::callback("拒絕並標記正常", format!("review:reject:{id}")),
        ]]);
        let text = notices::review_card(
            &case,
            "待審核舉報 · /spam",
            "受理：封禁並訓練。\n拒絕：標記為正常，並記錄舉報者被拒次數。",
            None,
        );
        let request = bot
            .send_message(ChatId(review_chat), text)
            .parse_mode(ParseMode::Html)
            .reply_markup(keyboard);
        let result = tokio::time::timeout(Duration::from_secs(30), async { request.await })
            .await
            .context("Telegram request timed out")
            .and_then(|r| r.map_err(anyhow::Error::from));
        let sent = match result {
            Ok(sent) => sent,
            Err(err) => {
                record_error(runtime, id, &err, guard).await?;
                return Ok(());
            }
        };
        let id = id.to_string();
        let save_guard = guard.clone();
        runtime.with_conn(move |conn| {
            let _guard = save_guard;
            conn.execute("UPDATE report_deliveries SET review_message_id=?2,last_error=NULL WHERE case_id=?1",params![id,sent.id.0])?;
            Ok(())
        }).await?;
    }
    if confirmation.is_none() {
        let text = if pending {
            "已送交舉報處理頻道審核。"
        } else {
            "舉報已有處理結果。"
        };
        let request = bot
            .send_message(ChatId(case.chat_id), text)
            .reply_parameters(
                teloxide::types::ReplyParameters::new(MessageId(command))
                    .allow_sending_without_reply(),
            );
        let result = tokio::time::timeout(Duration::from_secs(30), async { request.await })
            .await
            .context("Telegram request timed out")
            .and_then(|r| r.map_err(anyhow::Error::from));
        let sent = match result {
            Ok(sent) => sent,
            Err(err) => {
                record_error(runtime, id, &err, guard).await?;
                return Ok(());
            }
        };
        let id = id.to_string();
        let save_guard = guard.clone();
        runtime.with_conn(move |conn| {
            let _guard = save_guard;
            let tx = conn.transaction()?;
            remember_confirmation(&tx,&id,case.chat_id,sent.id.0)?;
            tx.execute("UPDATE report_deliveries SET confirmation_message_id=?2,state='done',last_error=NULL WHERE case_id=?1",params![id,sent.id.0])?;
            tx.commit()?;
            Ok(())
        }).await?;
    }
    Ok(())
}

async fn record_error(
    runtime: &Runtime,
    id: &str,
    error: &anyhow::Error,
    guard: Arc<tokio::sync::OwnedMutexGuard<()>>,
) -> Result<()> {
    if let Some(teloxide::RequestError::RetryAfter(delay)) =
        error.downcast_ref::<teloxide::RequestError>()
    {
        runtime.delay_telegram_queue(delay.seconds()).await?;
    }
    let id = id.to_string();
    let error = notices::diagnostic(&runtime.config, &error.to_string())
        .chars()
        .take(1000)
        .collect::<String>();
    runtime
        .with_conn(move |conn| {
            let _guard = guard;
            conn.execute(
                "UPDATE report_deliveries SET last_error=?2 WHERE case_id=?1",
                params![id, error],
            )?;
            Ok(())
        })
        .await
}
