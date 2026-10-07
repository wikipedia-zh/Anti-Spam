use super::*;
use rusqlite::OptionalExtension;

pub(super) struct ActionGuards {
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
    training_mode: String,
    trained: bool,
    review_sent: bool,
    audit_id: Option<i64>,
    audit_done: bool,
}

fn supported_action(action: &ActionKind) -> bool {
    matches!(
        action,
        ActionKind::AutoBan
            | ActionKind::GuestBotBan
            | ActionKind::GuestInvokerBan
            | ActionKind::SpamBan
            | ActionKind::ReportApproved
    )
}

pub(super) fn insert_origin_case(
    tx: &rusqlite::Transaction<'_>,
    case: &CaseRecord,
    header: &str,
) -> Result<bool> {
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
    Ok(inserted != 0)
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

    pub(super) async fn origin_guards(&self, case: &CaseRecord) -> Arc<ActionGuards> {
        Arc::new(ActionGuards {
            _review: self.review_guard(&case.id).await,
            _user: self.user_action_guard(case.target_user_id).await,
        })
    }

    async fn queue_origin_ban(&self, case: CaseRecord, header: String) -> Result<()> {
        self.queue_origin_bans(vec![(case, header)]).await
    }

    pub(super) async fn queue_origin_bans(&self, cases: Vec<(CaseRecord, String)>) -> Result<()> {
        self.queue_origin_bans_with_thresholds(
            cases
                .into_iter()
                .map(|(case, header)| (case, header, None))
                .collect(),
        )
        .await
    }

    pub(super) async fn queue_scored_ban(
        &self,
        case: CaseRecord,
        header: String,
        threshold: f64,
    ) -> Result<()> {
        anyhow::ensure!(
            case.action == ActionKind::AutoBan
                && case
                    .model_score
                    .is_some_and(|score| passes_threshold(score, threshold)),
            "invalid scored ban"
        );
        self.queue_origin_bans_with_thresholds(vec![(case, header, Some(threshold))])
            .await
    }

    async fn queue_origin_bans_with_thresholds(
        &self,
        cases: Vec<(CaseRecord, String, Option<f64>)>,
    ) -> Result<()> {
        anyhow::ensure!(
            cases.iter().all(|(case, _, _)| matches!(
                case.action,
                ActionKind::AutoBan | ActionKind::GuestBotBan | ActionKind::GuestInvokerBan
            )),
            "unsupported original ban action"
        );
        let mut ids: Vec<_> = cases.iter().map(|(c, _, _)| c.id.clone()).collect();
        ids.sort();
        ids.dedup();
        let mut review_guards = Vec::new();
        for id in ids {
            review_guards.push(self.review_guard(&id).await);
        }
        let mut users: Vec<_> = cases.iter().map(|(c, _, _)| c.target_user_id).collect();
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
            for (case, header, threshold) in cases {
                if insert_origin_case(&tx, &case, &header)? {
                    if let Some(value) = threshold {
                        case_thresholds::record(
                            &tx,
                            &case.id,
                            "detection",
                            value,
                            case.model_score.context("missing model score")?,
                        )?;
                    }
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
                "SELECT j.header,j.ban_done,j.delete_done,j.outcome_unknown,j.announced_status,j.broadcast_done,j.attempts,
                 COALESCE(f.training_mode,'none'),COALESCE(f.training_done,1),COALESCE(f.review_sent,1),f.audit_id,COALESCE(f.audit_done,1)
                 FROM origin_ban_jobs j LEFT JOIN ban_followups f ON f.case_id=j.case_id
                 WHERE j.case_id=?1 AND j.state='pending' AND j.next_attempt_at<=?2
                 AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?2",
                params![case.id,now], |r| Ok((OriginJob {
                    case: case.clone(), header:r.get(0)?, banned:r.get(1)?, deleted:r.get(2)?,
                    unknown:r.get(3)?, announced:r.get(4)?, broadcast:r.get(5)?, error:None,
                    training_mode:r.get(7)?,trained:r.get(8)?,review_sent:r.get(9)?,audit_id:r.get(10)?,audit_done:r.get(11)?,
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
                tx.execute("UPDATE cases SET status=?2,log_message_id=?3 WHERE id=?1 AND action IN ('auto_ban','guest_bot_ban','guest_invoker_ban','spam_ban','report_approved')
                    AND status NOT IN ('reversed','reversal_pending')",
                    params![job.case.id,job.case.status,job.case.log_message_id])?;
                if job.banned && job.trained && job.case.action == ActionKind::AutoBan
                    && job.case.matched_rule_pattern.as_deref() != Some("BOTSPAM") {
                    if let Some(score) = job.case.model_score {
                        let shared: bool = tx.query_row("SELECT netban_eligible FROM cases WHERE id=?1", [&job.case.id], |r| r.get(0))?;
                        if !shared {
                            case_thresholds::record(&tx, &job.case.id, "network", threshold, score)?;
                        }
                    }
                }
                if job.banned && job.trained && (matches!(job.training_mode.as_str(),"direct"|"report")
                    || netban_eligible(&job.case.action,job.case.model_score,threshold,job.case.matched_rule_pattern.as_deref())) {
                    network_delivery::enqueue_network_ban(&tx,&job.case.id,test_group)?;
                }
            }
            tx.commit()?;
            Ok(())
        }).await
    }

    async fn finish_ban_training(&self, job: &OriginJob, guards: Arc<ActionGuards>) -> Result<()> {
        let job = job.clone();
        let test_group = self.config.test_group_id;
        self.with_model_transaction(move |tx| {
            let guards = guards;
            if job.training_mode == "report"
                || (job.training_mode == "direct"
                    && job.case.matched_rule_pattern.as_deref() != Some("BOTSPAM"))
            {
                reliability::write_sample(tx, "spam", &job.case.evidence_text, Some(&job.case.id))?;
            }
            tx.execute(
                "UPDATE ban_followups SET training_done=1 WHERE case_id=?1",
                params![job.case.id],
            )?;
            if matches!(job.training_mode.as_str(), "direct" | "report") {
                network_delivery::enqueue_network_ban(tx, &job.case.id, test_group)?;
            }
            // Return the guard with the value so it survives the transaction
            // commit and model publication inside with_model_transaction.
            Ok(guards)
        })
        .await?;
        Ok(())
    }

    async fn finish_ban_followup(
        &self,
        id: &str,
        stage: &'static str,
        guards: Arc<ActionGuards>,
    ) -> Result<()> {
        let id = id.to_string();
        self.with_conn(move |conn| {
            let _guards = guards;
            anyhow::ensure!(
                matches!(stage, "review_sent" | "audit_done"),
                "invalid followup stage"
            );
            conn.execute(
                &format!("UPDATE ban_followups SET {stage}=1 WHERE case_id=?1"),
                params![id],
            )?;
            Ok(())
        })
        .await
    }
}

async fn api<T, E: Into<anyhow::Error>>(
    request: impl std::future::Future<Output = Result<T, E>>,
) -> Result<T> {
    tokio::time::timeout(Duration::from_secs(30), request)
        .await
        .context("Telegram request timed out")?
        .map_err(Into::into)
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

async fn policy_still_enabled(
    runtime: &Runtime,
    case: &CaseRecord,
    guards: Arc<ActionGuards>,
) -> Result<bool> {
    if case.matched_rule_pattern.as_deref() == Some("WARN") {
        return warning_queue::eligible(runtime, case).await;
    }
    if matches!(
        case.action,
        ActionKind::SpamBan | ActionKind::ReportApproved
    ) {
        return Ok(true);
    }
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
        let threshold = runtime.effective_threshold(Some(case.chat_id)).await?;
        let id = case.id.clone();
        runtime
            .with_conn(move |conn| {
                let _guards = guards;
                case_thresholds::record(conn, &id, "enforcement", threshold, score)
            })
            .await?;
        return Ok(passes_threshold(score, threshold));
    }
    let settings = runtime.get_group_modules(case.chat_id).await?;
    let rules = runtime.spam_rules.read().await;
    Ok(reason.split('；').any(|part| match part {
        "CONTACT" => settings.no_contact,
        "VOICE" => settings.no_voice,
        "EXEC_FILE" => settings.no_exec,
        "ARABIC" => settings.no_halal,
        part if part.starts_with("REGEX@") => part[6..].parse::<i64>().ok().is_some_and(|id| {
            rules.iter().any(|rule| {
                rule.id == id
                    && (regex_is_match(&rule.regex, &case.target_name)
                        || regex_is_match(&rule.regex, &case.evidence_text))
            })
        }),
        // Unstructured historical labels are not reinterpreted here.
        _ => true,
    }))
}

fn ban_status(job: &OriginJob) -> &'static str {
    if !job.banned {
        return "ban_failed";
    }
    if !job.deleted {
        return "banned_delete_failed";
    }
    match job.case.action {
        ActionKind::GuestBotBan => "guest_bot_banned",
        ActionKind::GuestInvokerBan => "guest_invoker_banned",
        ActionKind::ReportApproved => "approved_and_banned",
        ActionKind::SpamBan if job.training_mode == "direct" => "force_approved",
        ActionKind::SpamBan => "done",
        _ => "auto_banned",
    }
}

async fn ban_pending(
    bot: &Bot,
    runtime: &Runtime,
    job: &mut OriginJob,
    guards: Arc<ActionGuards>,
) -> Result<bool> {
    if job.banned {
        return Ok(true);
    }
    let previously_unknown = job.unknown;
    job.unknown = true;
    runtime
        .save_origin_job(job, "pending", guards.clone())
        .await?;
    let mut limited = false;
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
            limited = record_error(runtime, job, "ban", &err).await?;
        }
    }
    job.case.status = ban_status(job).to_string();
    runtime.save_origin_job(job, "pending", guards).await?;
    Ok(!limited)
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
    let previously_banned = supported_action(&case.action)
        && !matches!(
            case.status.as_str(),
            "ban_pending" | "ban_failed" | "reversed" | "reversal_pending"
        );
    let Some(mut job) = runtime.claim_origin_ban(case, guards.clone()).await? else {
        return Ok(previously_banned);
    };

    if !job.banned {
        let manual = matches!(
            job.case.action,
            ActionKind::SpamBan | ActionKind::ReportApproved
        );
        if manual {
            let actor = job.case.actor_user_id.context("missing approving actor")?;
            let allowed = if job.training_mode == "review"
                || job.case.matched_rule_pattern.as_deref() == Some("WARN")
            {
                match api(async {
                    bot.get_chat_member(ChatId(job.case.chat_id), UserId(actor as u64))
                        .await
                })
                .await
                {
                    Ok(member) => member.kind.is_privileged(),
                    Err(err) => {
                        record_error(runtime, &mut job, "check_actor", &err).await?;
                        runtime.save_origin_job(&job, "pending", guards).await?;
                        return Ok(false);
                    }
                }
            } else {
                runtime.can_review(actor).await
            };
            if !allowed {
                job.case.status = "ban_failed".into();
                job.error = Some(
                    "Approving actor no longer has permission; pending action cancelled".into(),
                );
                runtime.save_origin_job(&job, "cancelled", guards).await?;
                return Ok(false);
            }
        }
        let exempt = !policy_still_enabled(runtime, &job.case, guards.clone()).await?
            || runtime.is_maintainer(job.case.target_user_id).await
            || is_platform_pseudo_user(job.case.target_user_id)
            || runtime.is_group_banned(job.case.chat_id).await
            || (!manual
                && (runtime
                    .is_global_whitelisted(job.case.target_user_id)
                    .await?
                    || runtime
                        .is_group_whitelisted(job.case.chat_id, job.case.target_user_id)
                        .await?));
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

    let manual = matches!(
        job.case.action,
        ActionKind::SpamBan | ActionKind::ReportApproved
    );
    if manual && !ban_pending(bot, runtime, &mut job, guards.clone()).await? {
        return Ok(job.banned);
    }

    if !job.deleted && (!manual || job.banned) {
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

    if !manual && !ban_pending(bot, runtime, &mut job, guards.clone()).await? {
        return Ok(job.banned);
    }
    job.case.status = ban_status(&job).to_string();
    // Applying the ban and queuing its cross-group deliveries commit together.
    runtime
        .save_origin_job(&job, "pending", guards.clone())
        .await?;

    if job.banned && !job.trained {
        if let Err(err) = runtime.finish_ban_training(&job, guards.clone()).await {
            record_error(runtime, &mut job, "training", &err).await?;
            runtime.save_origin_job(&job, "pending", guards).await?;
            return Ok(true);
        }
        job.trained = true;
        runtime
            .save_origin_job(&job, "pending", guards.clone())
            .await?;
    }

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
        let result = if job.case.action == ActionKind::ReportApproved
            || (job.case.action == ActionKind::SpamBan && !job.banned)
        {
            Ok(Ok(()))
        } else {
            tokio::time::timeout(
                Duration::from_secs(30),
                notify_group(
                    bot,
                    runtime,
                    &job.case,
                    job.case.log_message_id.context("missing log message")?,
                    title,
                ),
            )
            .await
        };
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
        if let Err(err) = api(rule_notices::capture(bot, runtime, &job.case)).await {
            record_error(runtime, &mut job, "bot_spam_rules", &err).await?;
            runtime.save_origin_job(&job, "pending", guards).await?;
            return Ok(true);
        }
    }
    if job.banned && !job.review_sent {
        if job.training_mode == "review"
            && job.case.matched_rule_pattern.as_deref() != Some("BOTSPAM")
        {
            if let Err(err) = api(queue_training_review(
                bot,
                runtime,
                &job.case,
                guards.clone(),
            ))
            .await
            {
                record_error(runtime, &mut job, "training_review", &err).await?;
                runtime.save_origin_job(&job, "pending", guards).await?;
                return Ok(true);
            }
        }
        runtime
            .finish_ban_followup(&job.case.id, "review_sent", guards.clone())
            .await?;
        job.review_sent = true;
    }
    if job.banned && !job.audit_done {
        if let (Some(id), Some(chat)) = (job.audit_id, runtime.audit_log_chat().await) {
            let actor = job.case.actor_user_id.context("missing audit actor")?;
            let name = escape_html(job.case.actor_name.as_deref().unwrap_or(""));
            let command = if job.training_mode == "direct" {
                "/sb -f"
            } else {
                "/sb"
            };
            let text=format!("<b>維護操作 #{id}</b>\n<b>指令</b>: <code>{command}</code>\n<b>操作者</b>: {name} (<code>{actor}</code>)\n<b>群組</b>: <code>{}</code>\n<b>內容</b>: {} 對象={}\n復原：<code>/revert {id}</code>",job.case.chat_id,chinese_case_action(&job.case),job.case.target_user_id);
            if let Err(err) = api(async {
                bot.send_message(ChatId(chat), text)
                    .parse_mode(ParseMode::Html)
                    .await
            })
            .await
            {
                record_error(runtime, &mut job, "audit", &err).await?;
                runtime.save_origin_job(&job, "pending", guards).await?;
                return Ok(true);
            }
        }
        runtime
            .finish_ban_followup(&job.case.id, "audit_done", guards.clone())
            .await?;
        job.audit_done = true;
    }
    let done = job.banned
        && job.deleted
        && job.broadcast
        && job.announced == job.case.status
        && job.trained
        && job.review_sent
        && job.audit_done;
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

pub(super) async fn execute_scored_ban(
    bot: &Bot,
    runtime: &Runtime,
    case: CaseRecord,
    header: &str,
    threshold: f64,
) -> Result<bool> {
    runtime
        .queue_scored_ban(case.clone(), header.to_string(), threshold)
        .await?;
    let banned = attempt_origin_ban(bot, runtime, case.clone()).await?;
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
            if let Err(err) = rule_notices::retry(&bot, &runtime).await {
                log::warn!("rule notice queue: {err}");
            }
            if let Err(err) = warning_queue::retry(&bot, &runtime).await {
                log::warn!("warning queue: {err}");
            }
            if let Err(err) = report_delivery::retry(&bot, &runtime).await {
                log::warn!("report delivery queue: {err}");
            }
            if let Err(err) = moderation_queue::deliver_review_updates(&bot, &runtime, None).await {
                log::warn!("review notification queue: {err}");
            }
            if let Err(err) = restriction_retry::retry(&bot, &runtime).await {
                log::warn!("restriction queue: {err}");
            }
            sleep(Duration::from_secs(5)).await;
        }
    })
}
