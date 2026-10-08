use super::*;
use rusqlite::OptionalExtension;
use std::future::IntoFuture;

impl Runtime {
    pub(super) fn migrate_v35_to_v36(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch("
            CREATE TABLE IF NOT EXISTS network_catchups (
                chat_id INTEGER NOT NULL,message_id INTEGER NOT NULL,user_id INTEGER NOT NULL,
                case_id TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','done','cancelled')),
                delete_done INTEGER NOT NULL DEFAULT 0,notice_id INTEGER,cleanup_at INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,next_attempt_at INTEGER NOT NULL DEFAULT 0,last_error TEXT,
                PRIMARY KEY(chat_id,message_id,user_id));
            CREATE INDEX IF NOT EXISTS idx_network_catchup_due ON network_catchups(state,next_attempt_at);
            PRAGMA user_version=36;")?;
        tx.commit()?;
        Ok(())
    }

    pub(super) async fn queue_network_catchup(
        &self,
        case_id: &str,
        chat: i64,
        message: i32,
        user: i64,
    ) -> Result<bool> {
        if self.is_maintainer(user).await || is_platform_pseudo_user(user) {
            return Ok(false);
        }
        let guard = self.user_action_guard(user).await;
        let id = case_id.to_string();
        let test_group = self.config.test_group_id;
        self.with_conn(move|conn| {
            let _guard=guard;let tx=conn.transaction()?;
            if network_delivery::eligible_target(&tx,&id,chat,test_group)?!=Some(user) {return Ok(false);}
            let inserted=tx.execute("INSERT OR IGNORE INTO network_catchups(chat_id,message_id,user_id,case_id) VALUES (?1,?2,?3,?4)",params![chat,message,user,id])?;
            if inserted!=0 {
                // A fresh join or message can reveal that an earlier successful
                // ban no longer holds. Replays never reset a pending backoff.
                tx.execute("INSERT INTO network_deliveries(case_id,chat_id) VALUES (?1,?2)
                    ON CONFLICT(case_id,chat_id) DO UPDATE SET state='pending',attempts=0,next_attempt_at=0,last_error=NULL
                    WHERE network_deliveries.state!='pending'",params![id,chat])?;
            }
            tx.commit()?;Ok(true)
        }).await
    }
}

#[derive(Clone)]
struct Job {
    chat: i64,
    message: i32,
    user: i64,
    case_id: String,
    deleted: bool,
    notice: Option<i32>,
    cleanup: i64,
    access_revision: group_access::Observation,
}

async fn update(
    runtime: &Runtime,
    job: &Job,
    state: &str,
    error: Option<String>,
    guard: Arc<tokio::sync::OwnedMutexGuard<()>>,
) -> Result<()> {
    let job = job.clone();
    let state = state.to_string();
    runtime.with_conn(move|conn| {
        let _guard=guard;
        conn.execute("UPDATE network_catchups SET state=?4,delete_done=?5,notice_id=?6,cleanup_at=?7,last_error=?8,
            next_attempt_at=CASE WHEN ?8 IS NULL AND ?6 IS NOT NULL THEN ?7 ELSE next_attempt_at END
            WHERE chat_id=?1 AND message_id=?2 AND user_id=?3",params![job.chat,job.message,job.user,state,job.deleted,job.notice,job.cleanup,error])?;
        Ok(())
    }).await
}

async fn api<T>(
    request: impl std::future::Future<Output = ResponseResult<T>>,
) -> ResponseResult<T> {
    tokio::time::timeout(Duration::from_secs(30), request)
        .await
        .unwrap_or_else(|_| {
            Err(teloxide::RequestError::Io(
                std::io::Error::new(std::io::ErrorKind::TimedOut, "Telegram request timed out")
                    .into(),
            ))
        })
}

async fn failed(
    runtime: &Runtime,
    job: &Job,
    error: teloxide::RequestError,
    guard: Arc<tokio::sync::OwnedMutexGuard<()>>,
) -> Result<()> {
    if let teloxide::RequestError::RetryAfter(delay) = &error {
        runtime.delay_telegram_queue(delay.seconds()).await?;
    }
    runtime.record_group_access_error(job.chat,job.access_revision,&error).await?;
    update(
        runtime,
        job,
        "pending",
        Some(
            notices::diagnostic(&runtime.config, &error.to_string())
                .chars()
                .take(2000)
                .collect(),
        ),
        guard,
    )
    .await
}

async fn attempt(bot: &Bot, runtime: &Runtime, chat: i64, message: i32, user: i64) -> Result<()> {
    let guard = Arc::new(runtime.user_action_guard(user).await);
    let db_guard = guard.clone();
    let test = runtime.config.test_group_id;
    let claimed=runtime.with_conn(move|conn| {
        let _guard=db_guard;let tx=conn.transaction()?;let now=Utc::now().timestamp();
        if group_access::blocked(&tx,chat)? {return Ok(None);}
        let access_revision=group_access::observation(&tx,chat)?;
        let row=tx.query_row("SELECT x.case_id,x.delete_done,x.notice_id,x.cleanup_at,x.attempts,d.state
            FROM network_catchups x JOIN network_deliveries d ON d.case_id=x.case_id AND d.chat_id=x.chat_id
            WHERE x.chat_id=?1 AND x.message_id=?2 AND x.user_id=?3 AND x.state='pending' AND x.next_attempt_at<=?4
            AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?4",
            params![chat,message,user,now],|r|Ok((Job{chat,message,user,access_revision,case_id:r.get(0)?,deleted:r.get(1)?,notice:r.get(2)?,cleanup:r.get(3)?},r.get::<_,u32>(4)?,r.get::<_,String>(5)?))).optional()?;
        let Some((job,attempts,delivery))=row else{return Ok(None);};
        let eligible=network_delivery::eligible_target(&tx,&job.case_id,chat,test)?==Some(user);
        if delivery=="pending" && eligible && job.notice.is_none(){return Ok(None);}
        let attempt=attempts.saturating_add(1).min(31);let delay=(60_i64<<attempt.saturating_sub(1).min(6)).min(3600);
        tx.execute("UPDATE network_catchups SET attempts=?4,next_attempt_at=?5 WHERE chat_id=?1 AND message_id=?2 AND user_id=?3",params![chat,message,user,attempt,now+delay])?;
        tx.commit()?;Ok(Some((job,eligible && delivery=="done")))
    }).await?;
    let Some((mut job, active)) = claimed else {
        return Ok(());
    };
    let active = active && !runtime.is_maintainer(user).await && !is_platform_pseudo_user(user);
    // A notice already sent only needs cleanup, including after reversal.
    if let Some(notice) = job.notice {
        if active && job.cleanup > Utc::now().timestamp() {
            return update(runtime, &job, "pending", None, guard).await;
        }
        match api(bot
            .delete_message(ChatId(chat), MessageId(notice))
            .into_future())
        .await
        {
            Ok(_) => {}
            Err(e) if e.to_string().contains("message to delete not found") => {}
            Err(e) => return failed(runtime, &job, e, guard).await,
        }
        return update(
            runtime,
            &job,
            if active { "done" } else { "cancelled" },
            None,
            guard,
        )
        .await;
    }
    if !active {
        return update(runtime, &job, "cancelled", None, guard).await;
    }
    // Permissions may have changed since the ban was sent. A failed lookup
    // must not be treated as proof that the member is safe to moderate.
    match api(bot
        .get_chat_member(ChatId(chat), UserId(user as u64))
        .into_future())
    .await
    {
        Ok(member) if member.kind.is_privileged() => {
            return update(runtime, &job, "cancelled", None, guard).await
        }
        Ok(_) => {}
        Err(e) => return failed(runtime, &job, e, guard).await,
    }
    if !job.deleted {
        match api(bot
            .delete_message(ChatId(chat), MessageId(message))
            .into_future())
        .await
        {
            Ok(_) => {}
            Err(e) if e.to_string().contains("message to delete not found") => {}
            Err(e) => return failed(runtime, &job, e, guard).await,
        }
        job.deleted = true;
        update(runtime, &job, "pending", None, guard.clone()).await?;
    }
    let text = format!(
        "<b>已同步跨群封禁</b>\n<b>對象</b>: <code>{user}</code>\n<b>案例</b>: <code>{}</code>",
        escape_html(&job.case_id)
    );
    // Telegram cannot deduplicate a send whose response was lost. Once its
    // message ID is known, retries only finish deleting that notice.
    match api(bot
        .send_message(ChatId(chat), text)
        .parse_mode(ParseMode::Html)
        .into_future())
    .await
    {
        Ok(sent) => {
            job.notice = Some(sent.id.0);
            job.cleanup = Utc::now().timestamp() + 180;
            update(runtime, &job, "pending", None, guard).await
        }
        Err(e) => failed(runtime, &job, e, guard).await,
    }
}

pub(super) async fn retry(bot: &Bot, runtime: &Runtime) -> Result<()> {
    let keys=runtime.with_conn(|conn| {
        let mut stmt=conn.prepare("SELECT x.chat_id,x.message_id,x.user_id FROM network_catchups x
            JOIN network_deliveries d ON d.case_id=x.case_id AND d.chat_id=x.chat_id
            JOIN cases c ON c.id=x.case_id
            WHERE x.state='pending' AND x.next_attempt_at<=?1
            AND NOT EXISTS(SELECT 1 FROM group_access a WHERE a.chat_id=x.chat_id AND a.state IN ('left','unavailable'))
            AND (d.state!='pending' OR x.notice_id IS NOT NULL OR c.status IN ('reversed','reversal_pending'))
            AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?1
            ORDER BY x.next_attempt_at,x.chat_id,x.message_id,x.user_id LIMIT 20")?;
        let rows=stmt.query_map([Utc::now().timestamp()],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i32>(1)?,r.get::<_,i64>(2)?)))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }).await?;
    for (chat, message, user) in keys {
        attempt(bot, runtime, chat, message, user).await?;
    }
    Ok(())
}

pub(super) async fn observe(
    bot: &Bot,
    runtime: &Runtime,
    case: &CaseRecord,
    message: &Message,
    user: i64,
) -> Result<bool> {
    if !runtime
        .queue_network_catchup(&case.id, message.chat.id.0, message.id.0, user)
        .await?
    {
        return Ok(false);
    }
    // Once saved, a transient delivery failure must not fall through to a
    // second moderation path. The worker retains responsibility after restart.
    if let Err(err) = deliver_network_bans(bot, runtime, Some(&case.id)).await {
        log::warn!(
            "network catch-up ban: {}",
            notices::diagnostic(&runtime.config, &err.to_string())
        );
    }
    if let Err(err) = attempt(bot, runtime, message.chat.id.0, message.id.0, user).await {
        log::warn!(
            "network catch-up notice: {}",
            notices::diagnostic(&runtime.config, &err.to_string())
        );
    }
    Ok(true)
}
