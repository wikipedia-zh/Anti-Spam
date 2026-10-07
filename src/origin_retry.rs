use super::*;
use rusqlite::OptionalExtension;

struct ActionGuards {
    _review: tokio::sync::OwnedMutexGuard<()>,
    _user: tokio::sync::OwnedMutexGuard<()>,
}

#[derive(Clone)]
struct OriginJob {
    case: CaseRecord,
    header: String,
    banned: bool,
    deleted: bool,
    unknown: bool,
    announced: String,
    broadcast: bool,
    error: Option<String>,
}

fn supported_action(action: &ActionKind) -> bool {
    matches!(
        action,
        ActionKind::AutoBan | ActionKind::GuestBotBan | ActionKind::GuestInvokerBan
    )
}

impl Runtime {
    pub(super) fn migrate_v22_to_v23(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS origin_ban_jobs (
                case_id TEXT PRIMARY KEY,
                header TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','done','cancelled')),
                ban_done INTEGER NOT NULL DEFAULT 0,
                delete_done INTEGER NOT NULL DEFAULT 0,
                outcome_unknown INTEGER NOT NULL DEFAULT 0,
                announced_status TEXT NOT NULL DEFAULT '',
                broadcast_done INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at INTEGER NOT NULL DEFAULT 0,
                last_error TEXT
             );
             CREATE INDEX IF NOT EXISTS idx_origin_ban_due ON origin_ban_jobs(state,next_attempt_at);
             PRAGMA user_version=23;",
        )?;
        tx.commit()?;
        Ok(())
    }

    async fn origin_guards(&self, case: &CaseRecord) -> Arc<ActionGuards> {
        Arc::new(ActionGuards {
            _review: self.review_guard(&case.id).await,
            _user: self.user_action_guard(case.target_user_id).await,
        })
    }

    async fn queue_origin_ban(&self, case: CaseRecord, header: String) -> Result<()> {
        self.queue_origin_bans(vec![(case, header)]).await
    }

    pub(super) async fn queue_origin_bans(&self, cases: Vec<(CaseRecord, String)>) -> Result<()> {
        anyhow::ensure!(
            cases.iter().all(|(case, _)| supported_action(&case.action)),
            "unsupported original ban action"
        );
        let mut ids: Vec<_> = cases.iter().map(|(c, _)| c.id.clone()).collect();
        ids.sort();
        ids.dedup();
        let mut review_guards = Vec::new();
        for id in ids {
            review_guards.push(self.review_guard(&id).await);
        }
        let mut users: Vec<_> = cases.iter().map(|(c, _)| c.target_user_id).collect();
        users.sort();
        users.dedup();
        let mut user_guards = Vec::new();
        for id in users {
            user_guards.push(self.user_action_guard(id).await);
        }
        self.with_conn(move |conn| {
            // Keep the locks until the blocking transaction ends, even if the
            // caller is cancelled while SQLite is busy.
            let _guards = (review_guards, user_guards);
            let tx = conn.transaction()?;
            for (case, header) in cases {
                let guest = matches!(
                    case.action,
                    ActionKind::GuestBotBan | ActionKind::GuestInvokerBan
                );
                let inserted = tx.execute(
                    "INSERT OR IGNORE INTO cases (id,action,chat_id,target_user_id,target_name,
                 actor_user_id,actor_name,source_message_id,evidence_text,model_score,
                 matched_rule_id,matched_rule_pattern,status,log_message_id,created_at)
                 SELECT ?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,'ban_pending',NULL,?13
                 WHERE ?14=0 OR NOT EXISTS(SELECT 1 FROM cases WHERE (action=?2 OR (action='unbanned' AND matched_rule_pattern=?12)) AND chat_id=?3
                     AND target_user_id=?4 AND source_message_id=?8)",
                    params![
                        case.id,
                        case.action.as_str(),
                        case.chat_id,
                        case.target_user_id,
                        case.target_name,
                        case.actor_user_id,
                        case.actor_name,
                        case.source_message_id,
                        case.evidence_text,
                        case.model_score,
                        case.matched_rule_id,
                        case.matched_rule_pattern,
                        case.created_at.to_rfc3339(),
                        guest
                    ],
                )?;
                // Replays must not revive a reversed case or adopt a historical failure.
                if inserted != 0 {
                    tx.execute(
                        "INSERT INTO origin_ban_jobs(case_id,header,delete_done) VALUES (?1,?2,?3)",
                        params![case.id, header, case.source_message_id.is_none()],
                    )?;
                }
            }
            tx.commit()?;
            Ok(())
        })
        .await
    }

    async fn claim_origin_ban(
        &self,
        case: CaseRecord,
        guards: Arc<ActionGuards>,
    ) -> Result<Option<OriginJob>> {
        self.with_conn(move |conn| {
            let _guards = guards;
            let tx = conn.transaction()?;
            let now = Utc::now().timestamp();
            let job = tx.query_row(
                "SELECT header,ban_done,delete_done,outcome_unknown,announced_status,broadcast_done,attempts
                 FROM origin_ban_jobs WHERE case_id=?1 AND state='pending' AND next_attempt_at<=?2
                 AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?2",
                params![case.id,now], |r| Ok((OriginJob {
                    case: case.clone(), header:r.get(0)?, banned:r.get(1)?, deleted:r.get(2)?,
                    unknown:r.get(3)?, announced:r.get(4)?, broadcast:r.get(5)?, error:None,
                },r.get::<_,u32>(6)?)),
            ).optional()?;
            let Some((job,attempts)) = job else { return Ok(None); };
            if !supported_action(&case.action) || matches!(case.status.as_str(), "reversed" | "reversal_pending") {
                tx.execute("UPDATE origin_ban_jobs SET state='cancelled' WHERE case_id=?1",params![case.id])?;
                tx.commit()?;
                return Ok(None);
            }
            let attempt = attempts.saturating_add(1).min(31);
            let delay = (60_i64 << attempt.saturating_sub(1).min(6)).min(3600);
            tx.execute("UPDATE origin_ban_jobs SET attempts=?2,next_attempt_at=?3 WHERE case_id=?1",
                params![case.id,attempt,now+delay])?;
            tx.commit()?;
            Ok(Some(job))
        }).await
    }

    async fn save_origin_job(
        &self,
        job: &OriginJob,
        state: &str,
        guards: Arc<ActionGuards>,
    ) -> Result<()> {
        let job = job.clone();
        let state = state.to_string();
        let default_threshold = self.config.spam_threshold;
        let test_group = self.config.test_group_id;
        self.with_conn(move |conn| {
            let _guards = guards;
            let tx = conn.transaction()?;
            let threshold = Self::load_threshold(&tx)?.unwrap_or(default_threshold);
            let changed = tx.execute(
                "UPDATE origin_ban_jobs SET state=?2,ban_done=?3,delete_done=?4,outcome_unknown=?5,
                 announced_status=?6,broadcast_done=?7,last_error=?8 WHERE case_id=?1 AND state='pending'",
                params![job.case.id,state,job.banned,job.deleted,job.unknown,job.announced,job.broadcast,job.error],
            )?;
            if changed != 0 {
                tx.execute("UPDATE cases SET status=?2,log_message_id=?3 WHERE id=?1 AND action IN ('auto_ban','guest_bot_ban','guest_invoker_ban')
                    AND status NOT IN ('reversed','reversal_pending')",
                    params![job.case.id,job.case.status,job.case.log_message_id])?;
                if job.banned && netban_eligible(&job.case.action,job.case.model_score,threshold,job.case.matched_rule_pattern.as_deref()) {
                    network_delivery::enqueue_network_ban(&tx,&job.case.id,test_group)?;
                }
            }
            tx.commit()?;
            Ok(())
        }).await
    }
}

async fn api<T>(request: impl std::future::Future<Output = ResponseResult<T>>) -> Result<T> {
    Ok(tokio::time::timeout(Duration::from_secs(30), request)
        .await
        .context("Telegram request timed out")??)
}

async fn record_error(
    runtime: &Runtime,
    job: &mut OriginJob,
    stage: &str,
    error: &anyhow::Error,
) -> Result<bool> {
    let diagnostic = notices::diagnostic(&runtime.config, &error.to_string());
    log::warn!("case={} stage={stage}: {diagnostic}", job.case.id);
    job.error = Some(format!(
        "{stage}: {}",
        diagnostic.chars().take(1000).collect::<String>()
    ));
    if let Some(teloxide::RequestError::RetryAfter(delay)) =
        error.downcast_ref::<teloxide::RequestError>()
    {
        runtime.delay_telegram_queue(delay.seconds()).await?;
        return Ok(true);
    }
    Ok(false)
}

async fn policy_still_enabled(runtime: &Runtime, case: &CaseRecord) -> Result<bool> {
    if matches!(
        case.action,
        ActionKind::GuestBotBan | ActionKind::GuestInvokerBan
    ) {
        return Ok(runtime.get_group_modules(case.chat_id).await?.guest_ban);
    }
    let Some(reason) = case.matched_rule_pattern.as_deref() else {
        return Ok(true);
    };
    if let Some(score) = case.model_score {
        return Ok(passes_threshold(
            score,
            runtime.effective_threshold(Some(case.chat_id)).await?,
        ));
    }
    let settings = runtime.get_group_modules(case.chat_id).await?;
    let rules = runtime.spam_rules.read().await;
    Ok(reason.split('；').any(|part| match part {
        "CONTACT" => settings.no_contact,
        "VOICE" => settings.no_voice,
        "EXEC_FILE" => settings.no_exec,
        "ARABIC" => settings.no_halal,
        part if part.starts_with("REGEX@") => part[6..]
            .parse::<i64>()
            .ok()
            .is_some_and(|id| rules.iter().any(|rule| rule.id == id)),
        // Unstructured historical labels are not reinterpreted here.
        _ => true,
    }))
}

pub(super) async fn attempt_origin_ban(
    bot: &Bot,
    runtime: &Runtime,
    case: CaseRecord,
) -> Result<bool> {
    let guards = runtime.origin_guards(&case).await;
    let Some(case) = runtime.load_case(&case.id).await? else {
        return Ok(false);
    };
    let Some(mut job) = runtime.claim_origin_ban(case, guards.clone()).await? else {
        return Ok(false);
    };

    if !job.banned {
        let exempt = !policy_still_enabled(runtime, &job.case).await?
            || runtime.is_maintainer(job.case.target_user_id).await
            || is_platform_pseudo_user(job.case.target_user_id)
            || runtime.is_group_banned(job.case.chat_id).await
            || runtime
                .is_global_whitelisted(job.case.target_user_id)
                .await?
            || runtime
                .is_group_whitelisted(job.case.chat_id, job.case.target_user_id)
                .await?;
        let admin = if exempt {
            true
        } else {
            match api(async {
                bot.get_chat_member(
                    ChatId(job.case.chat_id),
                    UserId(job.case.target_user_id as u64),
                )
                .await
            })
            .await
            {
                Ok(member) => {
                    member.kind.is_privileged()
                        || (job.case.action == ActionKind::GuestBotBan
                            && !matches!(
                                member.kind,
                                teloxide::types::ChatMemberKind::Left
                                    | teloxide::types::ChatMemberKind::Banned(_)
                            ))
                }
                Err(err) => {
                    record_error(runtime, &mut job, "check_member", &err).await?;
                    runtime.save_origin_job(&job, "pending", guards).await?;
                    return Ok(false);
                }
            }
        };
        if admin {
            job.case.status = "ban_failed".into();
            job.error = Some("Target is exempt or policy changed; pending action cancelled".into());
            runtime.save_origin_job(&job, "cancelled", guards).await?;
            return Ok(false);
        }
    }

    if !job.deleted {
        let id = job
            .case
            .source_message_id
            .context("missing source message")?;
        let result = api(async {
            bot.delete_message(ChatId(job.case.chat_id), MessageId(id))
                .await
        })
        .await;
        match result {
            Ok(_) => job.deleted = true,
            Err(err) if err.to_string().contains("message to delete not found") => {
                job.deleted = true
            }
            Err(err) => {
                if record_error(runtime, &mut job, "delete_message", &err).await? {
                    runtime.save_origin_job(&job, "pending", guards).await?;
                    return Ok(job.banned);
                }
            }
        }
        runtime
            .save_origin_job(&job, "pending", guards.clone())
            .await?;
    }

    if !job.banned {
        let previously_unknown = job.unknown;
        job.unknown = true;
        runtime
            .save_origin_job(&job, "pending", guards.clone())
            .await?;
        match api(async {
            bot.ban_chat_member(
                ChatId(job.case.chat_id),
                UserId(job.case.target_user_id as u64),
            )
            .await
        })
        .await
        {
            Ok(_) => {
                job.banned = true;
                job.unknown = false;
            }
            Err(err) => {
                let definite_failure = matches!(
                    err.downcast_ref::<teloxide::RequestError>(),
                    Some(
                        teloxide::RequestError::Api(_)
                            | teloxide::RequestError::MigrateToChatId(_)
                            | teloxide::RequestError::RetryAfter(_)
                    )
                );
                job.unknown = previously_unknown || !definite_failure;
                job.case.status = "ban_failed".into();
                if record_error(runtime, &mut job, "ban", &err).await? {
                    runtime.save_origin_job(&job, "pending", guards).await?;
                    return Ok(false);
                }
            }
        }
    }
    job.case.status = match (job.banned, job.deleted) {
        (true, true) => match job.case.action {
            ActionKind::GuestBotBan => "guest_bot_banned",
            ActionKind::GuestInvokerBan => "guest_invoker_banned",
            _ => "auto_banned",
        },
        (true, false) => "banned_delete_failed",
        (false, _) => "ban_failed",
    }
    .into();
    // Applying the ban and queuing its cross-group deliveries commit together.
    runtime
        .save_origin_job(&job, "pending", guards.clone())
        .await?;

    if job.announced != job.case.status {
        let log_result = if let Some(id) = job.case.log_message_id {
            api(async {
                bot.edit_message_text(
                    ChatId(runtime.config.log_channel_id),
                    MessageId(id),
                    notices::action_log(&job.case),
                )
                .parse_mode(ParseMode::Html)
                .await
            })
            .await
            .map(|_| id)
        } else {
            api(log_action(bot, runtime, &job.case)).await
        };
        match log_result {
            Ok(id) => job.case.log_message_id = Some(id),
            Err(err) if err.to_string().contains("message is not modified") => {}
            Err(err) if err.to_string().contains("message to edit not found") => {
                job.case.log_message_id = None;
                record_error(runtime, &mut job, "log", &err).await?;
                runtime.save_origin_job(&job, "pending", guards).await?;
                return Ok(job.banned);
            }
            Err(err) => {
                record_error(runtime, &mut job, "log", &err).await?;
                runtime.save_origin_job(&job, "pending", guards).await?;
                return Ok(job.banned);
            }
        }
        runtime
            .save_origin_job(&job, "pending", guards.clone())
            .await?;
        let title = match (job.banned, job.deleted) {
            (false, _) => "<b>封禁失敗，稍後重試</b>",
            (true, false) => "<b>已封禁，訊息刪除稍後重試</b>",
            _ => &job.header,
        };
        // sendMessage has no idempotency key. An interrupted acknowledgement
        // may duplicate a notice; known log IDs are reused on ordinary retries.
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            notify_group(
                bot,
                runtime,
                &job.case,
                job.case.log_message_id.context("missing log message")?,
                title,
            ),
        )
        .await;
        match result {
            Ok(Ok(())) => job.announced = job.case.status.clone(),
            result => {
                let err = match result {
                    Ok(Err(err)) => err,
                    _ => anyhow::anyhow!("Telegram request timed out"),
                };
                record_error(runtime, &mut job, "notice", &err).await?;
                runtime.save_origin_job(&job, "pending", guards).await?;
                return Ok(job.banned);
            }
        }
        runtime
            .save_origin_job(&job, "pending", guards.clone())
            .await?;
    }
    if job.banned && !job.broadcast {
        if let Some(chat) = runtime.exchange_channel().await {
            let result = api(try_send_exchange_message(
                bot,
                chat,
                "report",
                "bad",
                serde_json::json!({"id":job.case.target_user_id,"is_banned":true}),
            ))
            .await;
            if let Err(err) = result {
                record_error(runtime, &mut job, "broadcast", &err).await?;
                runtime.save_origin_job(&job, "pending", guards).await?;
                return Ok(true);
            }
        }
        job.broadcast = true;
        runtime
            .save_origin_job(&job, "pending", guards.clone())
            .await?;
    }
    if job.banned && job.case.matched_rule_pattern.as_deref() == Some("BOTSPAM") {
        capture_bot_spam_rules(bot, runtime, job.case.chat_id, &job.case.evidence_text).await;
    }
    let done = job.banned && job.deleted && job.broadcast && job.announced == job.case.status;
    if done {
        job.error = None;
    }
    runtime
        .save_origin_job(&job, if done { "done" } else { "pending" }, guards)
        .await?;
    Ok(job.banned)
}

pub(super) async fn execute_auto_ban(
    bot: &Bot,
    runtime: &Runtime,
    case: CaseRecord,
    header: &str,
) -> Result<bool> {
    runtime
        .queue_origin_ban(case.clone(), header.to_string())
        .await?;
    let banned = attempt_origin_ban(bot, runtime, case.clone()).await?;
    // The per-user action lock must be released before draining network jobs.
    deliver_network_bans(bot, runtime, Some(&case.id)).await?;
    Ok(banned)
}

pub(super) async fn retry_origin_bans(bot: &Bot, runtime: &Runtime) -> Result<usize> {
    let pending = runtime.with_conn(|conn| {
        let mut stmt = conn.prepare("SELECT case_id FROM origin_ban_jobs WHERE state='pending' AND next_attempt_at<=?1
            AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?1 ORDER BY next_attempt_at,case_id LIMIT 20")?;
        let rows = stmt.query_map(params![Utc::now().timestamp()],|r| r.get::<_,String>(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }).await?;
    let count = pending.len();
    for id in pending {
        if let Some(case) = runtime.load_case(&id).await? {
            if let Err(err) = attempt_origin_ban(bot, runtime, case).await {
                log::warn!(
                    "original ban {id}: {}",
                    notices::diagnostic(&runtime.config, &err.to_string())
                );
            }
        }
    }
    Ok(count)
}

pub(super) fn spawn_origin_worker(bot: Bot, runtime: Arc<Runtime>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            if let Err(err) = retry_origin_bans(&bot, &runtime).await {
                log::warn!("original ban queue: {err}");
            }
            sleep(Duration::from_secs(5)).await;
        }
    })
}
