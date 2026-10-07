use super::*;
use rusqlite::OptionalExtension;

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct Warning {
    chat: i64,
    command: i32,
    target: i64,
    count: i64,
    settings: WarnSettings,
    case_id: Option<String>,
    source: Option<i32>,
    text: String,
    step: String,
    notice: Option<i32>,
    cleanup_at: Option<i64>,
}

impl Runtime {
    pub(super) fn migrate_v27_to_v28(conn: &mut Connection) -> Result<()> {
        conn.execute_batch("BEGIN;
            CREATE TABLE IF NOT EXISTS warning_requests (
                chat_id INTEGER NOT NULL,message_id INTEGER NOT NULL,target_id INTEGER NOT NULL,
                case_id TEXT,payload TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,next_attempt_at INTEGER NOT NULL DEFAULT 0,last_error TEXT,
                PRIMARY KEY(chat_id,message_id));
            CREATE INDEX IF NOT EXISTS idx_warning_due ON warning_requests(state,next_attempt_at);
            CREATE TABLE IF NOT EXISTS warning_removals (
                chat_id INTEGER NOT NULL,message_id INTEGER NOT NULL,removed INTEGER NOT NULL,remaining INTEGER NOT NULL,
                PRIMARY KEY(chat_id,message_id));
            PRAGMA user_version=28; COMMIT;")?;
        Ok(())
    }

    pub(super) async fn queue_warning(
        &self,
        mut case: CaseRecord,
        command: i32,
        reason: String,
        source: Option<i32>,
    ) -> Result<Warning> {
        let guard = self.user_action_guard(case.target_user_id).await;
        self.with_conn(move |conn| {
            let _guard = guard;
            let tx = conn.transaction()?;
            let prior = tx.query_row("SELECT payload FROM warning_requests WHERE chat_id=?1 AND message_id=?2",
                params![case.chat_id,command], |r|r.get::<_,String>(0)).optional()?;
            if let Some(payload) = prior { return Ok(serde_json::from_str(&payload)?); }
            let settings = tx.query_row("SELECT threshold,action,action_duration_secs,ot_warn_count,ot_template FROM group_warn_settings WHERE chat_id=?1",
                [case.chat_id],|r|Ok(WarnSettings{threshold:r.get(0)?,action:r.get(1)?,action_duration_secs:r.get(2)?,ot_warn_count:r.get(3)?,ot_template:r.get(4)?})).optional()?.unwrap_or_default();
            let amount = if source.is_some() {settings.ot_warn_count.max(1)} else {1};
            for _ in 0..amount {
                tx.execute("INSERT INTO warns(chat_id,user_id,reason,warned_by,created_at) VALUES (?1,?2,?3,?4,?5)",
                    params![case.chat_id,case.target_user_id,(!reason.is_empty()).then_some(&reason),case.actor_user_id,case.created_at.to_rfc3339()])?;
            }
            let count: i64 = tx.query_row("SELECT COUNT(*) FROM warns WHERE chat_id=?1 AND user_id=?2",params![case.chat_id,case.target_user_id],|r|r.get(0))?;
            let case_id = if count >= settings.threshold {
                case.action = match settings.action.as_str() {"ban"=>ActionKind::SpamBan,"kick"=>ActionKind::Kick,_=>ActionKind::Mute};
                case.evidence_text = format!("累計警告達 {count} 次（門檻 {}）",settings.threshold);
                case.matched_rule_pattern = Some("WARN".into());
                case.source_message_id = None;
                let header = "<b>警告達門檻，已自動處置</b>";
                if case.action == ActionKind::SpamBan {
                    anyhow::ensure!(origin_retry::insert_origin_case(&tx,&case,header)?,"case already exists");
                    tx.execute("INSERT INTO moderation_requests(chat_id,message_id,case_id) VALUES (?1,?2,?3)",params![case.chat_id,command,case.id])?;
                } else {
                    let until = settings.action_duration_secs.filter(|s|*s>0).map(|s|case.created_at.timestamp().checked_add(s).filter(|t|DateTime::from_timestamp(*t,0).is_some()).context("invalid mute duration")).transpose()?;
                    restriction_retry::insert_restriction(&tx,&case,command,if case.action==ActionKind::Mute {until} else {None},header.into())?;
                }
                Some(case.id.clone())
            } else {None};
            let text = if source.is_some() {
                settings.ot_template.clone().unwrap_or_else(default_ot_template)
                    .replace("{user}",&mention_link(case.target_user_id,&case.target_name)).replace("{count}",&count.to_string())
            } else {
                let reason_line = if reason.is_empty() {String::new()} else {format!("\n原因：{}",escape_html(&reason))};
                format!("{} 已被警告，目前累計 {count} 次（門檻 {}）。{reason_line}",mention_link(case.target_user_id,&case.target_name),settings.threshold)
            };
            let warning = Warning{chat:case.chat_id,command,target:case.target_user_id,count,settings,case_id,source,text,
                step:if source.is_some(){"delete"}else{"notice"}.into(),notice:None,cleanup_at:None};
            tx.execute("INSERT INTO warning_requests(chat_id,message_id,target_id,case_id,payload) VALUES (?1,?2,?3,?4,?5)",
                params![warning.chat,command,warning.target,warning.case_id,serde_json::to_string(&warning)?])?;
            tx.commit()?;
            Ok(warning)
        }).await
    }

    pub(super) async fn remove_warning_request(
        &self,
        chat: i64,
        target: i64,
        amount: i64,
        command: i32,
    ) -> Result<(i64, i64)> {
        let guard = self.user_action_guard(target).await;
        self.with_conn(move |conn| {
            let _guard=guard; let tx=conn.transaction()?;
            if let Some(result)=tx.query_row("SELECT removed,remaining FROM warning_removals WHERE chat_id=?1 AND message_id=?2",params![chat,command],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?))).optional()? {return Ok(result);}
            let removed=tx.execute("DELETE FROM warns WHERE id IN (SELECT id FROM warns WHERE chat_id=?1 AND user_id=?2 ORDER BY created_at DESC,id DESC LIMIT ?3)",params![chat,target,amount.max(1)])? as i64;
            let remaining=tx.query_row("SELECT COUNT(*) FROM warns WHERE chat_id=?1 AND user_id=?2",params![chat,target],|r|r.get::<_,i64>(0))?;
            // A later warning must not revive an earlier, withdrawn request.
            tx.execute("UPDATE restriction_jobs SET state='cancelled' WHERE state='pending' AND json_extract(payload,'$.step')='apply' AND json_extract(payload,'$.uncertain')=0
                AND case_id IN (SELECT case_id FROM warning_requests WHERE chat_id=?1 AND target_id=?2 AND json_extract(payload,'$.settings.threshold')>?3)",params![chat,target,remaining])?;
            tx.execute("UPDATE origin_ban_jobs SET state='cancelled' WHERE state='pending' AND ban_done=0 AND outcome_unknown=0
                AND case_id IN (SELECT case_id FROM warning_requests WHERE chat_id=?1 AND target_id=?2 AND json_extract(payload,'$.settings.threshold')>?3)",params![chat,target,remaining])?;
            tx.execute("UPDATE cases SET status='action_cancelled' WHERE id IN (SELECT case_id FROM warning_requests WHERE chat_id=?1 AND target_id=?2)
                AND (id IN (SELECT case_id FROM restriction_jobs WHERE state='cancelled' AND json_extract(payload,'$.step')='apply' AND json_extract(payload,'$.uncertain')=0)
                OR id IN (SELECT case_id FROM origin_ban_jobs WHERE state='cancelled' AND ban_done=0 AND outcome_unknown=0))",params![chat,target])?;
            tx.execute("INSERT INTO warning_removals(chat_id,message_id,removed,remaining) VALUES (?1,?2,?3,?4)",params![chat,command,removed,remaining])?;
            tx.commit()?;Ok((removed,remaining))
        }).await
    }
}

pub(super) async fn eligible(runtime: &Runtime, case: &CaseRecord) -> Result<bool> {
    let id = case.id.clone();
    let saved = runtime
        .with_conn(move |conn| {
            Ok(conn
                .query_row(
                    "SELECT payload FROM warning_requests WHERE case_id=?1",
                    [id],
                    |r| r.get::<_, String>(0),
                )
                .optional()?)
        })
        .await?;
    let Some(saved) = saved else {
        return Ok(false);
    };
    let warning: Warning = serde_json::from_str(&saved)?;
    let current = runtime.get_warn_settings(case.chat_id).await?;
    Ok(runtime
        .warn_count(case.chat_id, case.target_user_id)
        .await?
        >= current.threshold
        && current.action == warning.settings.action
        && current.action_duration_secs == warning.settings.action_duration_secs)
}

async fn telegram<T>(
    runtime: &Runtime,
    future: impl std::future::Future<Output = std::result::Result<T, teloxide::RequestError>>,
) -> Result<T> {
    let result = tokio::time::timeout(Duration::from_secs(30), future)
        .await
        .context("Telegram request timed out")?;
    if let Err(teloxide::RequestError::RetryAfter(delay)) = &result {
        runtime.delay_telegram_queue(delay.seconds()).await?;
    }
    Ok(result?)
}

async fn save(
    runtime: &Runtime,
    warning: &Warning,
    state: &str,
    guard: Arc<tokio::sync::OwnedMutexGuard<()>>,
) -> Result<()> {
    let warning = warning.clone();
    let state = state.to_string();
    runtime.with_conn(move |conn| {
        let _guard = guard;
        conn.execute("UPDATE warning_requests SET payload=?3,state=?4,last_error=NULL,next_attempt_at=?5 WHERE chat_id=?1 AND message_id=?2",
            params![warning.chat,warning.command,serde_json::to_string(&warning)?,state,if warning.step=="cleanup" {warning.cleanup_at.unwrap_or(Utc::now().timestamp()+60)} else {Utc::now().timestamp()+60}])?;
        Ok(())
    }).await
}

async fn delete(bot: &Bot, runtime: &Runtime, chat: i64, message: i32) -> Result<()> {
    match telegram(runtime, async {
        bot.delete_message(ChatId(chat), MessageId(message)).await
    })
    .await
    {
        Ok(_) => Ok(()),
        Err(e)
            if e.to_string()
                .to_lowercase()
                .contains("message to delete not found") =>
        {
            Ok(())
        }
        Err(e) => Err(e),
    }
}

async fn deliver(
    bot: &Bot,
    runtime: &Runtime,
    warning: &mut Warning,
    guard: Arc<tokio::sync::OwnedMutexGuard<()>>,
) -> Result<()> {
    if warning.step == "delete" {
        delete(
            bot,
            runtime,
            warning.chat,
            warning.source.context("missing source")?,
        )
        .await?;
        warning.step = "notice".into();
        save(runtime, warning, "pending", guard.clone()).await?;
    }
    if warning.step == "notice" {
        let (text, buttons) = if warning.source.is_some() {
            extract_template_buttons(&warning.text)
        } else {
            (warning.text.clone(), None)
        };
        let sent = telegram(runtime, async {
            let req = bot
                .send_message(ChatId(warning.chat), text)
                .parse_mode(ParseMode::Html);
            if let Some(buttons) = buttons {
                req.reply_markup(buttons).await
            } else {
                req.await
            }
        })
        .await?;
        warning.notice = Some(sent.id.0);
        warning.cleanup_at = warning.source.map(|_| Utc::now().timestamp() + 86400);
        warning.step = "command".into();
        // Do not delay command deletion until the notice expires.
        save(runtime, warning, "pending", guard.clone()).await?;
    }
    if warning.step == "command" {
        delete(bot, runtime, warning.chat, warning.command).await?;
        if warning.source.is_some() {
            warning.cleanup_at = Some(warning.cleanup_at.unwrap_or(Utc::now().timestamp() + 86400));
            warning.step = "cleanup".into();
            save(runtime, warning, "pending", guard.clone()).await?;
        } else {
            warning.step = "done".into();
            save(runtime, warning, "done", guard.clone()).await?;
        }
    }
    if warning.step == "cleanup"
        && warning
            .cleanup_at
            .is_some_and(|t| t <= Utc::now().timestamp())
    {
        delete(
            bot,
            runtime,
            warning.chat,
            warning.notice.context("missing notice")?,
        )
        .await?;
        warning.step = "done".into();
        save(runtime, warning, "done", guard).await?;
    }
    Ok(())
}

pub(super) async fn attempt(bot: &Bot, runtime: &Runtime, chat: i64, command: i32) -> Result<()> {
    let guard = Arc::new(
        runtime
            .review_guard(&format!("warning:{chat}:{command}"))
            .await,
    );
    let claim_guard = guard.clone();
    let payload = runtime.with_conn(move |conn| {
        let _guard = claim_guard; let tx=conn.transaction()?; let now=Utc::now().timestamp();
        let row = tx.query_row("SELECT payload,attempts FROM warning_requests WHERE chat_id=?1 AND message_id=?2 AND state='pending' AND next_attempt_at<=?3 AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?3",
            params![chat,command,now],|r|Ok((r.get::<_,String>(0)?,r.get::<_,u32>(1)?))).optional()?;
        if let Some((_,attempts)) = &row {
            let n=attempts.saturating_add(1).min(31);
            tx.execute("UPDATE warning_requests SET attempts=?3,next_attempt_at=?4 WHERE chat_id=?1 AND message_id=?2",params![chat,command,n,now+(60_i64<<n.saturating_sub(1).min(6)).min(3600)])?;
        }
        tx.commit()?; Ok(row.map(|r|r.0))
    }).await?;
    let Some(payload) = payload else {
        return Ok(());
    };
    let mut warning: Warning = serde_json::from_str(&payload)?;
    if let Err(error) = deliver(bot, runtime, &mut warning, guard.clone()).await {
        let error = notices::diagnostic(&runtime.config, &error.to_string())
            .chars()
            .take(1000)
            .collect::<String>();
        log::warn!("warning delivery {chat}/{command}: {error}");
        runtime
            .with_conn(move |conn| {
                let _guard = guard;
                conn.execute(
                    "UPDATE warning_requests SET last_error=?3 WHERE chat_id=?1 AND message_id=?2",
                    params![chat, command, error],
                )?;
                Ok(())
            })
            .await?;
    }
    Ok(())
}

pub(super) async fn retry(bot: &Bot, runtime: &Runtime) -> Result<()> {
    let rows = runtime.with_conn(|conn| {
        let mut stmt=conn.prepare("SELECT chat_id,message_id FROM warning_requests WHERE state='pending' AND next_attempt_at<=?1 AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?1 ORDER BY next_attempt_at,chat_id,message_id LIMIT 20")?;
        let rows=stmt.query_map([Utc::now().timestamp()],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i32>(1)?)))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }).await?;
    for (chat, command) in rows {
        attempt(bot, runtime, chat, command).await?;
    }
    Ok(())
}

pub(super) async fn handle(
    bot: &Bot,
    runtime: &Runtime,
    message: &Message,
    target: i64,
    name: String,
    reason: String,
    source: Option<i32>,
) -> ResponseResult<()> {
    let Some(actor) = message.from.as_ref() else {
        return Ok(());
    };
    let case = CaseRecord {
        id: Uuid::new_v4().to_string(),
        action: ActionKind::Mute,
        chat_id: message.chat.id.0,
        target_user_id: target,
        target_name: name,
        actor_user_id: Some(actor.id.0 as i64),
        actor_name: Some(short_user(actor)),
        source_message_id: None,
        evidence_text: String::new(),
        model_score: None,
        matched_rule_id: None,
        matched_rule_pattern: None,
        status: "action_pending".into(),
        log_message_id: None,
        created_at: Utc::now(),
    };
    let warning = match runtime
        .queue_warning(case, message.id.0, reason, source)
        .await
    {
        Ok(warning) => warning,
        Err(error) => {
            log::error!(
                "save warning: {}",
                notices::diagnostic(&runtime.config, &error.to_string())
            );
            reply_ephemeral(bot, message, "未能儲存警告，請稍後重試。").await?;
            return Ok(());
        }
    };
    if let Some(id) = warning.case_id.as_ref() {
        if let Ok(Some(case)) = runtime.load_case(id).await {
            if case.action == ActionKind::SpamBan {
                let _ = origin_retry::attempt_origin_ban(bot, runtime, case).await;
            } else {
                let _ = restriction_retry::attempt(bot, runtime, case).await;
            }
        }
    }
    if let Err(error) = attempt(bot, runtime, warning.chat, warning.command).await {
        log::warn!("warning queue: {error}");
    }
    Ok(())
}
