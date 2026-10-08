use super::*;
use rusqlite::OptionalExtension;

pub(super) fn eligible_target(
    conn: &Connection,
    case_id: &str,
    chat_id: i64,
    test_group: Option<i64>,
) -> Result<Option<i64>> {
    Ok(conn.query_row("SELECT c.target_user_id FROM cases c JOIN group_module_settings g ON g.chat_id=?2
        WHERE c.id=?1 AND c.netban_eligible=1 AND g.netban=1
        AND (?3 IS NULL OR g.chat_id!=?3)
        AND c.action IN ('auto_ban','spam_ban','report_approved','guest_bot_ban','guest_invoker_ban')
        AND c.status NOT IN ('ban_pending','ban_failed','reversal_pending','reversed')
        AND NOT EXISTS(SELECT 1 FROM global_whitelist WHERE user_id=c.target_user_id)
        AND NOT EXISTS(SELECT 1 FROM group_whitelist WHERE chat_id=g.chat_id AND user_id=c.target_user_id)
        AND NOT EXISTS(SELECT 1 FROM banned_groups WHERE chat_id=g.chat_id)",
        params![case_id,chat_id,test_group], |r| r.get(0)).optional()?)
}

// The blacklist entry and its deliveries belong to the same transaction.
pub(super) fn enqueue_network_ban(
    tx: &rusqlite::Transaction<'_>,
    case_id: &str,
    test_group_id: Option<i64>,
) -> Result<()> {
    tx.execute(
        "UPDATE cases SET netban_eligible=1 WHERE id=?1
         AND action IN ('auto_ban','spam_ban','report_approved','guest_bot_ban','guest_invoker_ban')
         AND status NOT IN ('ban_pending','ban_failed','reversal_pending','reversed')
         AND NOT EXISTS (SELECT 1 FROM global_whitelist WHERE user_id=cases.target_user_id)",
        params![case_id],
    )?;
    tx.execute(
        "INSERT OR IGNORE INTO network_deliveries(case_id,chat_id)
         SELECT c.id,g.chat_id FROM cases c CROSS JOIN group_module_settings g
         WHERE c.id=?1 AND c.netban_eligible=1 AND g.netban=1 AND g.chat_id!=c.chat_id
         AND (?2 IS NULL OR g.chat_id!=?2)
         AND c.action IN ('auto_ban','spam_ban','report_approved','guest_bot_ban','guest_invoker_ban')
         AND c.status NOT IN ('ban_pending','ban_failed','reversal_pending','reversed')
         AND NOT EXISTS (SELECT 1 FROM global_whitelist WHERE user_id=c.target_user_id)
         AND NOT EXISTS (SELECT 1 FROM group_whitelist WHERE chat_id=g.chat_id AND user_id=c.target_user_id)
         AND NOT EXISTS (SELECT 1 FROM banned_groups WHERE chat_id=g.chat_id)
         AND NOT EXISTS (SELECT 1 FROM group_access WHERE chat_id=g.chat_id AND state IN ('left','unavailable'))
         AND NOT EXISTS (SELECT 1 FROM network_ban_targets WHERE case_id=c.id AND chat_id=g.chat_id)",
        params![case_id, test_group_id],
    )?;
    Ok(())
}

impl Runtime {
    pub(super) fn migrate_v19_to_v20(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS network_deliveries (
                case_id TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','done','cancelled')),
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at INTEGER NOT NULL DEFAULT 0,
                outcome_unknown INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                PRIMARY KEY(case_id,chat_id)
             );
             CREATE INDEX IF NOT EXISTS idx_network_delivery_due ON network_deliveries(state,next_attempt_at);
             CREATE TABLE IF NOT EXISTS telegram_retry_state (
                id INTEGER PRIMARY KEY CHECK(id=1),
                not_before INTEGER NOT NULL DEFAULT 0
             );
             INSERT OR IGNORE INTO telegram_retry_state(id) VALUES (1);
             UPDATE telegram_retry_state SET not_before=MAX(not_before,
                (SELECT not_before FROM reversal_retry_state WHERE id=1)) WHERE id=1;
             DROP TABLE reversal_retry_state;
             PRAGMA user_version=20;",
        )?;
        tx.commit()?;
        Ok(())
    }

    pub(super) async fn user_action_guard(&self, user_id: i64) -> tokio::sync::OwnedMutexGuard<()> {
        let lock = {
            let mut locks = self.user_action_locks.lock().await;
            locks.retain(|_, lock| lock.strong_count() > 0);
            let lock = locks
                .get(&user_id)
                .and_then(std::sync::Weak::upgrade)
                .unwrap_or_else(|| Arc::new(Mutex::new(())));
            locks.insert(user_id, Arc::downgrade(&lock));
            lock
        };
        lock.lock_owned().await
    }

    pub(super) async fn enqueue_network_deliveries(&self, case_id: &str) -> Result<()> {
        let case_id = case_id.to_string();
        let test_group_id = self.config.test_group_id;
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            enqueue_network_ban(&tx, &case_id, test_group_id)?;
            tx.commit()?;
            Ok(())
        })
        .await
    }

    pub(super) async fn delay_telegram_queue(&self, seconds: u32) -> Result<()> {
        let until = Utc::now().timestamp() + i64::from(seconds).max(1);
        self.with_conn(move |conn| {
            conn.execute(
                "UPDATE telegram_retry_state SET not_before=MAX(not_before,?1) WHERE id=1",
                params![until],
            )?;
            Ok(())
        })
        .await
    }

    async fn claim_network_delivery(
        &self,
        case_id: &str,
        chat_id: i64,
    ) -> Result<Option<(i64, bool, group_access::Observation)>> {
        let case_id = case_id.to_string();
        let test_group_id = self.config.test_group_id;
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            let now = Utc::now().timestamp();
            if operations::controls(&tx)?.network_paused {return Ok(None);}
            if group_access::blocked(&tx,chat_id)? {return Ok(None);}
            let revision=group_access::observation(&tx,chat_id)?;
            let due: Option<(u32,bool)> = tx.query_row(
                "SELECT attempts,outcome_unknown FROM network_deliveries WHERE case_id=?1 AND chat_id=?2
                 AND state='pending' AND next_attempt_at<=?3
                 AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?3",
                params![case_id,chat_id,now], |r| Ok((r.get(0)?,r.get(1)?)),
            ).optional()?;
            let Some((attempts,previously_uncertain)) = due else { return Ok(None); };
            let user_id = eligible_target(&tx, &case_id, chat_id, test_group_id)?;
            if user_id.is_some() {
                let attempt = attempts.saturating_add(1).min(31);
                let delay = (60_i64 << attempt.saturating_sub(1).min(6)).min(3600);
                tx.execute(
                    "UPDATE network_deliveries SET attempts=?3,next_attempt_at=?4,outcome_unknown=1
                     WHERE case_id=?1 AND chat_id=?2",
                    params![case_id,chat_id,attempt,now+delay],
                )?;
            } else {
                tx.execute("UPDATE network_deliveries SET state='cancelled' WHERE case_id=?1 AND chat_id=?2", params![case_id,chat_id])?;
            }
            tx.commit()?;
            Ok(user_id.map(|id| (id,previously_uncertain,revision)))
        }).await
    }

    async fn finish_network_delivery(&self, case_id: &str, chat_id: i64) -> Result<()> {
        let case_id = case_id.to_string();
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            tx.execute(
                "INSERT OR IGNORE INTO network_ban_targets(case_id,chat_id,created_at) VALUES (?1,?2,?3)",
                params![case_id,chat_id,Utc::now().to_rfc3339()],
            )?;
            tx.execute("UPDATE network_deliveries SET state='done',outcome_unknown=0,last_error=NULL WHERE case_id=?1 AND chat_id=?2", params![case_id,chat_id])?;
            tx.commit()?;
            Ok(())
        }).await
    }

    async fn cancel_network_delivery(
        &self,
        case_id: &str,
        chat_id: i64,
        uncertain: bool,
    ) -> Result<()> {
        let id = case_id.to_string();
        self.with_conn(move|conn| {
            conn.execute("UPDATE network_deliveries SET state='cancelled',outcome_unknown=?3,last_error='Target is exempt' WHERE case_id=?1 AND chat_id=?2",params![id,chat_id,uncertain])?;
            Ok(())
        }).await
    }

    async fn fail_network_delivery(
        &self,
        case_id: &str,
        chat_id: i64,
        error: &str,
        uncertain: bool,
    ) -> Result<()> {
        let case_id = case_id.to_string();
        let error = notices::diagnostic(&self.config, error)
            .chars()
            .take(2000)
            .collect::<String>();
        self.with_conn(move |conn| {
            conn.execute(
                "UPDATE network_deliveries SET last_error=?3,outcome_unknown=?4 WHERE case_id=?1 AND chat_id=?2",
                params![case_id,chat_id,error,uncertain],
            )?;
            Ok(())
        }).await
    }
}

pub(super) async fn deliver_network_bans(
    bot: &Bot,
    runtime: &Runtime,
    case_id: Option<&str>,
) -> Result<usize> {
    let filter = case_id.map(str::to_string);
    let pending = runtime.with_conn(move |conn| {
        let mut stmt = conn.prepare(
            "SELECT d.case_id,d.chat_id,c.target_user_id FROM network_deliveries d JOIN cases c ON c.id=d.case_id
             WHERE d.state='pending' AND d.next_attempt_at<=?1
             AND NOT EXISTS(SELECT 1 FROM group_access a WHERE a.chat_id=d.chat_id AND a.state IN ('left','unavailable'))
             AND (SELECT network_paused FROM operations_controls WHERE id=1)=0
             AND (?2 IS NULL OR d.case_id=?2)
             AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?1
             ORDER BY d.next_attempt_at,d.case_id,d.chat_id LIMIT 20",
        )?;
        let rows = stmt.query_map(params![Utc::now().timestamp(),filter], |r| Ok((r.get::<_,String>(0)?,r.get::<_,i64>(1)?,r.get::<_,i64>(2)?)))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }).await?;
    let mut attempted = 0;
    for (case_id, chat_id, target_user_id) in pending {
        let _guard = runtime.user_action_guard(target_user_id).await;
        let Some((user_id, previously_uncertain, access_revision)) =
            runtime.claim_network_delivery(&case_id, chat_id).await?
        else {
            continue;
        };
        attempted += 1;
        if runtime.is_maintainer(user_id).await || is_platform_pseudo_user(user_id) {
            runtime
                .cancel_network_delivery(&case_id, chat_id, previously_uncertain)
                .await?;
            continue;
        }
        match tokio::time::timeout(Duration::from_secs(30), async {
            bot.get_chat_member(ChatId(chat_id), UserId(user_id as u64))
                .await
        })
        .await
        {
            Ok(Ok(member)) if member.kind.is_privileged() => {
                runtime
                    .cancel_network_delivery(&case_id, chat_id, previously_uncertain)
                    .await?;
                continue;
            }
            Ok(Ok(_)) => {}
            result => {
                let error = match result {
                    Ok(Err(err)) => {
                        runtime.record_group_access_error(chat_id,access_revision,&err).await?;
                        if let teloxide::RequestError::RetryAfter(delay) = &err {
                            runtime.delay_telegram_queue(delay.seconds()).await?;
                        }
                        notices::diagnostic(&runtime.config, &err.to_string())
                    }
                    _ => "Telegram member lookup timed out".into(),
                };
                runtime
                    .fail_network_delivery(&case_id, chat_id, &error, previously_uncertain)
                    .await?;
                continue;
            }
        }
        let request = bot.ban_chat_member(ChatId(chat_id), UserId(user_id as u64));
        match tokio::time::timeout(Duration::from_secs(30), async { request.await }).await {
            Ok(Ok(_)) => runtime.finish_network_delivery(&case_id, chat_id).await?,
            Ok(Err(teloxide::RequestError::RetryAfter(delay))) => {
                runtime.delay_telegram_queue(delay.seconds()).await?;
                runtime
                    .fail_network_delivery(
                        &case_id,
                        chat_id,
                        "Telegram rate limit",
                        previously_uncertain,
                    )
                    .await?;
                break;
            }
            Ok(Err(err)) => {
                runtime.record_group_access_error(chat_id,access_revision,&err).await?;
                let uncertain = previously_uncertain
                    || !matches!(
                        err,
                        teloxide::RequestError::Api(_) | teloxide::RequestError::MigrateToChatId(_)
                    );
                runtime
                    .fail_network_delivery(&case_id, chat_id, &err.to_string(), uncertain)
                    .await?;
                log::warn!(
                    "network ban {case_id} to {chat_id} failed: {}",
                    notices::diagnostic(&runtime.config, &err.to_string())
                );
            }
            Err(_) => {
                runtime
                    .fail_network_delivery(&case_id, chat_id, "Telegram request timed out", true)
                    .await?
            }
        }
    }
    Ok(attempted)
}

pub(super) fn spawn_network_worker(bot: Bot, runtime: Arc<Runtime>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            if let Err(err) = group_access::reconcile(&bot, &runtime).await {
                log::warn!("group access check: {}", notices::diagnostic(&runtime.config, &err.to_string()));
            }
            if let Err(err) = deliver_network_bans(&bot, &runtime, None).await {
                log::warn!(
                    "network delivery queue: {}",
                    notices::diagnostic(&runtime.config, &err.to_string())
                );
            }
            if let Err(err) = network_catchup::retry(&bot, &runtime).await {
                log::warn!(
                    "network catch-up queue: {}",
                    notices::diagnostic(&runtime.config, &err.to_string())
                );
            }
            sleep(Duration::from_secs(5)).await;
        }
    })
}
