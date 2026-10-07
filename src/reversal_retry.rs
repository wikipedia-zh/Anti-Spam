use super::*;

fn retry_delay(attempt: u32) -> i64 {
    (60_i64 << attempt.saturating_sub(1).min(6)).min(3600)
}

impl Runtime {
    pub(super) fn migrate_v18_to_v19(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS reversal_retries (
                case_id TEXT PRIMARY KEY,
                actor_id INTEGER,
                actor_name TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at INTEGER NOT NULL DEFAULT 0,
                last_error TEXT
             );
             CREATE INDEX IF NOT EXISTS idx_reversal_retry_due
                ON reversal_retries(next_attempt_at);
             CREATE TABLE IF NOT EXISTS reversal_retry_state (
                id INTEGER PRIMARY KEY CHECK(id=1),
                not_before INTEGER NOT NULL DEFAULT 0
             );
             INSERT OR IGNORE INTO reversal_retry_state(id) VALUES (1);
             INSERT OR IGNORE INTO reversal_retries(case_id)
                SELECT id FROM cases WHERE status='reversal_pending';
             PRAGMA user_version=19;",
        )?;
        tx.commit()?;
        Ok(())
    }

    async fn begin_reversal(
        &self,
        case: &CaseRecord,
        actor_id: i64,
        actor_name: &str,
    ) -> Result<()> {
        let case = case.clone();
        let actor_name = actor_name.to_string();
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            if case.status != "reversal_pending" {
                tx.execute(
                    "INSERT OR IGNORE INTO network_ban_targets(case_id,chat_id,created_at) VALUES (?1,?2,?3)",
                    params![case.id, case.chat_id, Utc::now().to_rfc3339()],
                )?;
            }
            // A lost API response may still have applied the ban. Keep those
            // targets for reversal before cancelling unfinished deliveries.
            tx.execute(
                "INSERT OR IGNORE INTO network_ban_targets(case_id,chat_id,created_at)
                 SELECT case_id,chat_id,?2 FROM network_deliveries WHERE case_id=?1 AND outcome_unknown=1",
                params![case.id, Utc::now().to_rfc3339()],
            )?;
            tx.execute("UPDATE network_deliveries SET state='cancelled',outcome_unknown=0 WHERE case_id=?1 AND state!='done'", params![case.id])?;
            tx.execute("UPDATE cases SET status='reversal_pending' WHERE id=?1", params![case.id])?;
            tx.execute(
                "INSERT INTO reversal_retries(case_id,actor_id,actor_name) VALUES (?1,?2,?3)
                 ON CONFLICT(case_id) DO UPDATE SET
                    actor_id=COALESCE(reversal_retries.actor_id,excluded.actor_id),
                    actor_name=COALESCE(reversal_retries.actor_name,excluded.actor_name)",
                params![case.id, actor_id, actor_name],
            )?;
            tx.commit()?;
            Ok(())
        }).await
    }

    async fn claim_reversal_attempt(&self, case_id: &str) -> Result<(Option<i64>, Option<String>)> {
        let case_id = case_id.to_string();
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            let now = Utc::now().timestamp();
            let not_before: i64 = tx.query_row(
                "SELECT not_before FROM telegram_retry_state WHERE id=1",
                [],
                |r| r.get(0),
            )?;
            anyhow::ensure!(
                not_before <= now,
                "Telegram 暫時限制請求，系統會稍後自動重試"
            );
            let (attempts, actor_id, actor_name): (u32, Option<i64>, Option<String>) = tx
                .query_row(
                    "SELECT attempts,actor_id,actor_name FROM reversal_retries WHERE case_id=?1",
                    params![case_id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )?;
            let attempt = attempts.saturating_add(1).min(31);
            // Reserve the next attempt before the API call, including crash recovery.
            tx.execute(
                "UPDATE reversal_retries SET attempts=?2,next_attempt_at=?3 WHERE case_id=?1",
                params![case_id, attempt, now + retry_delay(attempt)],
            )?;
            tx.commit()?;
            Ok((actor_id, actor_name))
        })
        .await
    }

    async fn finish_reversal(&self, case: &CaseRecord) -> Result<()> {
        let case = case.clone();
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            tx.execute(
                "UPDATE cases SET action='unbanned',status='reversed',actor_user_id=?2,actor_name=?3 WHERE id=?1",
                params![case.id, case.actor_user_id, case.actor_name],
            )?;
            tx.execute("DELETE FROM reversal_retries WHERE case_id=?1", params![case.id])?;
            tx.commit()?;
            Ok(())
        }).await
    }

    async fn reversal_due(&self, case_id: &str) -> Result<bool> {
        let case_id = case_id.to_string();
        self.with_conn(move |conn| {
            Ok(conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM reversal_retries r JOIN cases c ON c.id=r.case_id
                 WHERE r.case_id=?1 AND c.status='reversal_pending' AND r.next_attempt_at<=?2
                 AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?2)",
                params![case_id, Utc::now().timestamp()],
                |r| r.get(0),
            )?)
        })
        .await
    }
}

pub(super) async fn reverse_ban_case(
    bot: &Bot,
    runtime: &Runtime,
    case: CaseRecord,
    actor_id: i64,
    actor_name: &str,
) -> Result<String, String> {
    let _guard = runtime.review_lock.lock().await;
    let _action_guard = runtime.user_action_guard(case.target_user_id).await;
    let case = runtime
        .load_case(&case.id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "案例不存在".to_string())?;
    if case.status == "reversed" {
        return Ok("此案例已撤銷".to_string());
    }
    runtime
        .begin_reversal(&case, actor_id, actor_name)
        .await
        .map_err(|e| format!("無法保存解封請求，請稍後重試：{e}"))?;
    attempt_reversal(bot, runtime, case).await
}

// Call with review_lock held so a manual retry cannot overlap the worker.
async fn attempt_reversal(
    bot: &Bot,
    runtime: &Runtime,
    case: CaseRecord,
) -> Result<String, String> {
    let actor = runtime
        .claim_reversal_attempt(&case.id)
        .await
        .map_err(|e| e.to_string())?;
    let result = apply_reversal(bot, runtime, case.clone(), actor).await;
    if let Err(error) = &result {
        let case_id = case.id.clone();
        let error = error.chars().take(2000).collect::<String>();
        if let Err(err) = runtime
            .with_conn(move |conn| {
                conn.execute(
                    "UPDATE reversal_retries SET last_error=?2 WHERE case_id=?1",
                    params![case_id, error],
                )?;
                Ok(())
            })
            .await
        {
            log::warn!("could not save reversal error for {}: {err}", case.id);
        }
    }
    result
}

async fn apply_reversal(
    bot: &Bot,
    runtime: &Runtime,
    mut case: CaseRecord,
    actor: (Option<i64>, Option<String>),
) -> Result<String, String> {
    let removed = runtime
        .purge_training_by_case(&case.id)
        .await
        .map_err(|e| e.to_string())?;
    runtime.rebuild_model().await.map_err(|e| e.to_string())?;
    let targets = runtime
        .list_network_ban_targets(&case.id)
        .await
        .map_err(|e| e.to_string())?;
    let mut errors = Vec::new();
    let mut retained = 0;
    for chat_id in targets {
        if runtime
            .has_other_ban_in_chat(&case.id, chat_id, case.target_user_id)
            .await
            .map_err(|e| e.to_string())?
        {
            retained += 1;
        } else {
            let request = bot
                .unban_chat_member(ChatId(chat_id), UserId(case.target_user_id as u64))
                .only_if_banned(true);
            match tokio::time::timeout(Duration::from_secs(30), async { request.await }).await {
                Ok(Ok(_)) => {}
                Ok(Err(teloxide::RequestError::RetryAfter(delay))) => {
                    runtime
                        .delay_telegram_queue(delay.seconds())
                        .await
                        .map_err(|e| e.to_string())?;
                    errors.push(format!("Telegram 要求等待 {} 秒", delay.seconds()));
                    break;
                }
                Ok(Err(err)) if unban_noop_reason(&err).is_some() => {}
                Ok(Err(err)) => {
                    errors.push(format!("群組 {chat_id}: {err}"));
                    continue;
                }
                Err(_) => {
                    errors.push(format!("群組 {chat_id}: 請求逾時"));
                    continue;
                }
            }
        }
        runtime
            .remove_network_ban_target(&case.id, chat_id)
            .await
            .map_err(|e| e.to_string())?;
    }
    if !errors.is_empty() {
        return Err(format!(
            "案例 {} 撤銷尚未完成；系統會自動重試失敗項目。{}",
            case.id,
            errors.join("；")
        ));
    }
    case.action = ActionKind::Unbanned;
    case.status = "reversed".to_string();
    case.actor_user_id = actor.0;
    case.actor_name = actor.1;
    runtime
        .finish_reversal(&case)
        .await
        .map_err(|e| e.to_string())?;
    if let Ok(id) = log_action(bot, runtime, &case).await {
        case.log_message_id = Some(id);
        if let Err(err) = runtime.persist_case(&case).await {
            log::warn!("could not save reversal log for {}: {err}", case.id);
        }
        let _ = notify_group(bot, runtime, &case, id, "<b>封禁案例已撤銷</b>").await;
    }
    broadcast_unban_if_fully_clear(bot, runtime, case.target_user_id).await;
    Ok(format!("已撤銷 case <code>{}</code>、移除 {removed} 筆訓練樣本；{retained} 個群組因其他有效案件而保留限制。", case.id))
}

pub(super) async fn retry_due_reversals(bot: &Bot, runtime: &Runtime) -> Result<usize> {
    let case_ids = runtime.with_conn(|conn| {
        conn.execute("DELETE FROM reversal_retries WHERE NOT EXISTS (SELECT 1 FROM cases WHERE id=case_id AND status='reversal_pending')", [])?;
        let mut stmt = conn.prepare(
            "SELECT case_id FROM reversal_retries WHERE next_attempt_at<=?1
             AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?1
             ORDER BY next_attempt_at,case_id LIMIT 10",
        )?;
        let rows = stmt.query_map(params![Utc::now().timestamp()], |r| r.get::<_, String>(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }).await?;
    let mut attempted = 0;
    for case_id in case_ids {
        let _guard = runtime.review_lock.lock().await;
        if !runtime.reversal_due(&case_id).await? {
            continue;
        }
        if let Some(case) = runtime.load_case(&case_id).await? {
            let _action_guard = runtime.user_action_guard(case.target_user_id).await;
            attempted += 1;
            if let Err(err) = attempt_reversal(bot, runtime, case).await {
                log::warn!("reversal retry {case_id}: {err}");
            }
        }
    }
    Ok(attempted)
}

pub(super) fn spawn_reversal_worker(
    bot: Bot,
    runtime: Arc<Runtime>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            if let Err(err) = retry_due_reversals(&bot, &runtime).await {
                log::warn!("reversal queue: {err}");
            }
            sleep(Duration::from_secs(10)).await;
        }
    })
}
