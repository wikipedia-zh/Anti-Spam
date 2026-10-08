use super::*;
use rusqlite::OptionalExtension;
use teloxide::types::{ChatMemberKind, ChatPermissions, UntilDate};

const ANSWER_SECONDS: i64 = 120;

#[derive(Clone, Serialize, Deserialize)]
struct Challenge {
    #[serde(default)]
    control_epoch: i64,
    chat_id: i64,
    user_id: i64,
    join_message_id: i32,
    name: String,
    a: u8,
    b: u8,
    deadline: i64,
    restrict_until: i64,
    prior_restrict_until: Option<i64>,
    question: Option<i32>,
    state: String,
    kick_until: Option<i64>,
    kick_started: bool,
}

impl Runtime {
    async fn captcha_control_epoch(&self) -> Result<i64> {
        self.with_conn(|conn| {
            Ok(conn.query_row(
                "SELECT captcha_epoch FROM operations_controls WHERE id=1",
                [],
                |r| r.get(0),
            )?)
        })
        .await
    }
    async fn queue_captcha(
        &self,
        mut job: Challenge,
        guard: Arc<tokio::sync::OwnedMutexGuard<()>>,
    ) -> Result<Option<Challenge>> {
        self.with_conn(move|conn| {
            let _guard=guard;let tx=conn.transaction()?;let controls=operations::controls(&tx)?;
            if controls.automatic_new_paused || controls.automatic_pending_paused {return Ok(None);}
            job.control_epoch=tx.query_row("SELECT captcha_epoch FROM operations_controls WHERE id=1",[],|r|r.get(0))?;
            tx.execute("INSERT INTO captcha_jobs(chat_id,user_id,payload,next_attempt_at) VALUES (?1,?2,?3,0) ON CONFLICT(chat_id,user_id) DO UPDATE SET payload=excluded.payload,next_attempt_at=0",params![job.chat_id,job.user_id,serde_json::to_string(&job)?])?;
            tx.commit()?;Ok(Some(job))
        }).await
    }
    pub(super) fn migrate_v20_to_v21(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS captcha_jobs (
                chat_id INTEGER NOT NULL, user_id INTEGER NOT NULL, payload TEXT NOT NULL,
                next_attempt_at INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT, PRIMARY KEY(chat_id,user_id)
             );
             CREATE INDEX IF NOT EXISTS idx_captcha_due ON captcha_jobs(next_attempt_at);
             PRAGMA user_version=21;",
        )?;
        tx.commit()?;
        Ok(())
    }

    async fn captcha_job(&self, chat_id: i64, user_id: i64) -> Result<Option<Challenge>> {
        self.with_conn(move |conn| {
            let raw: Option<String> = conn
                .query_row(
                    "SELECT payload FROM captcha_jobs WHERE chat_id=?1 AND user_id=?2",
                    params![chat_id, user_id],
                    |r| r.get(0),
                )
                .optional()?;
            raw.map(|s| serde_json::from_str(&s).map_err(Into::into))
                .transpose()
        })
        .await
    }

    async fn save_captcha(&self, job: &Challenge, next_attempt: i64) -> Result<()> {
        let job = job.clone();
        let payload = serde_json::to_string(&job)?;
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO captcha_jobs(chat_id,user_id,payload,next_attempt_at) VALUES (?1,?2,?3,?4)
                 ON CONFLICT(chat_id,user_id) DO UPDATE SET payload=excluded.payload,next_attempt_at=excluded.next_attempt_at",
                params![job.chat_id,job.user_id,payload,next_attempt],
            )?;
            Ok(())
        }).await
    }

    async fn claim_captcha(&self, chat_id: i64, user_id: i64) -> Result<bool> {
        self.with_conn(move |conn| {
            let now = Utc::now().timestamp();
            Ok(conn.execute(
                "UPDATE captcha_jobs SET attempts=MIN(attempts+1,10),next_attempt_at=?3+MIN(5*(1<<MIN(attempts,4)),60)
                 WHERE chat_id=?1 AND user_id=?2 AND next_attempt_at<=?3
                 AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?3",
                params![chat_id,user_id,now],
            )? != 0)
        }).await
    }

    async fn delete_captcha(&self, chat_id: i64, user_id: i64) -> Result<()> {
        self.with_conn(move |conn| {
            conn.execute(
                "DELETE FROM captcha_jobs WHERE chat_id=?1 AND user_id=?2",
                params![chat_id, user_id],
            )?;
            Ok(())
        })
        .await
    }
}

async fn telegram<T>(
    runtime: &Runtime,
    request: impl std::future::Future<Output = ResponseResult<T>>,
) -> Result<T> {
    match tokio::time::timeout(Duration::from_secs(20), request).await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(teloxide::RequestError::RetryAfter(delay))) => {
            runtime.delay_telegram_queue(delay.seconds()).await?;
            Err(teloxide::RequestError::RetryAfter(delay).into())
        }
        Ok(Err(err)) => Err(err.into()),
        Err(_) => anyhow::bail!("Telegram request timed out"),
    }
}

fn owns_restriction(member: &ChatMemberKind, until: i64) -> bool {
    let ChatMemberKind::Restricted(r) = member else {
        return false;
    };
    r.until_date == UntilDate::Date(DateTime::from_timestamp(until, 0).unwrap())
        && r.is_member
        && r.can_send_messages
        && !r.can_send_audios
        && !r.can_send_documents
        && !r.can_send_photos
        && !r.can_send_videos
        && !r.can_send_video_notes
        && !r.can_send_voice_notes
        && !r.can_send_other_messages
        && !r.can_add_web_page_previews
        && !r.can_change_info
        && !r.can_invite_users
        && !r.can_pin_messages
        && !r.can_manage_topics
        && !r.can_send_polls
}

fn owns_kick(member: &ChatMemberKind, until: Option<i64>) -> bool {
    matches!((member,until), (ChatMemberKind::Banned(b),Some(date))
        if b.until_date == UntilDate::Date(DateTime::from_timestamp(date,0).unwrap()))
}

async fn independent_ban(runtime: &Runtime, job: &Challenge) -> Result<bool> {
    runtime
        .has_other_ban_in_chat("", job.chat_id, job.user_id)
        .await
}

// The caller holds user_action_guard across state changes and API calls.
async fn run_captcha(bot: &Bot, runtime: &Runtime, mut job: Challenge) -> Result<()> {
    let chat = ChatId(job.chat_id);
    let user = UserId(job.user_id as u64);
    for _ in 0..6 {
        let previous_state = job.state.clone();
        if job.control_epoch != runtime.captcha_control_epoch().await?
            && !job.kick_started
            && matches!(job.state.as_str(), "prepare" | "waiting" | "kick")
        {
            job.state = "release".into();
        }
        if job.state == "waiting" && Utc::now().timestamp() < job.deadline {
            return Ok(());
        }
        if job.state == "waiting" {
            job.state = "kick".into();
        }
        if !matches!(job.state.as_str(), "unkick" | "cleanup") {
            let settings = runtime.get_group_modules(job.chat_id).await?;
            if !settings.captcha
                || runtime.is_group_banned(job.chat_id).await
                || runtime.config.test_group_id == Some(job.chat_id)
                || runtime.is_maintainer(job.user_id).await
                || runtime.is_global_whitelisted(job.user_id).await?
                || runtime
                    .is_group_whitelisted(job.chat_id, job.user_id)
                    .await?
            {
                job.state = "release".into();
            }
            if independent_ban(runtime, &job).await? {
                job.state = "cleanup".into();
            }
        }
        if job.state != previous_state {
            runtime.save_captcha(&job, 0).await?;
        }
        match job.state.as_str() {
            "prepare" => {
                // A question that was never delivered is not a failed answer.
                if Utc::now().timestamp() >= job.deadline
                    || job.restrict_until - Utc::now().timestamp() < 40
                {
                    job.state = "release".into();
                    runtime.save_captcha(&job, 0).await?;
                    continue;
                }
                let member =
                    telegram(runtime, async { bot.get_chat_member(chat, user).await }).await?;
                if !matches!(member.kind, ChatMemberKind::Member(_))
                    && !owns_restriction(&member.kind, job.restrict_until)
                    && !job
                        .prior_restrict_until
                        .is_some_and(|until| owns_restriction(&member.kind, until))
                {
                    job.state = "cleanup".into();
                    runtime.save_captcha(&job, 0).await?;
                    continue;
                }
                if !owns_restriction(&member.kind, job.restrict_until) {
                    job.restrict_until = Utc::now().timestamp() + ANSWER_SECONDS + 60;
                    runtime
                        .save_captcha(&job, Utc::now().timestamp() + 30)
                        .await?;
                    telegram(runtime, async {
                        bot.restrict_chat_member(chat, user, ChatPermissions::SEND_MESSAGES)
                            .use_independent_chat_permissions(true)
                            .until_date(DateTime::from_timestamp(job.restrict_until, 0).unwrap())
                            .await
                    })
                    .await?;
                }
                let text = format!("{}（<code>{}</code>）你好，請在 120 秒內回覆純數字答案，逾時將被移出群組：\n\n<b>{} + {} = ?</b>",
                    mention_link(job.user_id,&job.name),job.user_id,job.a,job.b);
                let sent = telegram(runtime, async {
                    bot.send_message(chat, text)
                        .parse_mode(ParseMode::Html)
                        .await
                })
                .await?;
                job.question = Some(sent.id.0);
                job.deadline = Utc::now().timestamp() + ANSWER_SECONDS;
                job.state = "waiting".into();
                runtime.save_captcha(&job, job.deadline).await?;
                return Ok(());
            }
            "release" => {
                let member =
                    telegram(runtime, async { bot.get_chat_member(chat, user).await }).await?;
                if (owns_restriction(&member.kind, job.restrict_until)
                    || job
                        .prior_restrict_until
                        .is_some_and(|until| owns_restriction(&member.kind, until)))
                    && !independent_ban(runtime, &job).await?
                {
                    telegram(runtime, async {
                        bot.restrict_chat_member(chat, user, ChatPermissions::all())
                            .await
                    })
                    .await?;
                }
                job.state = "cleanup".into();
                runtime.save_captcha(&job, 0).await?;
            }
            "kick" => {
                let member =
                    telegram(runtime, async { bot.get_chat_member(chat, user).await }).await?;
                if job.kick_started {
                    // Reconcile an interrupted kick; never kick a rejoined user again.
                    job.state = if owns_kick(&member.kind, job.kick_until) {
                        "unkick"
                    } else {
                        "cleanup"
                    }
                    .into();
                    runtime.save_captcha(&job, 0).await?;
                    continue;
                }
                if !matches!(member.kind, ChatMemberKind::Member(_))
                    && !owns_restriction(&member.kind, job.restrict_until)
                {
                    job.state = "cleanup".into();
                    runtime.save_captcha(&job, 0).await?;
                    continue;
                }
                job.kick_until = Some(Utc::now().timestamp() + 90);
                job.kick_started = true;
                runtime
                    .save_captcha(&job, Utc::now().timestamp() + 30)
                    .await?;
                let result = telegram(runtime, async {
                    bot.ban_chat_member(chat, user)
                        .until_date(DateTime::from_timestamp(job.kick_until.unwrap(), 0).unwrap())
                        .await
                })
                .await;
                if let Err(err) = result {
                    if matches!(
                        err.downcast_ref::<teloxide::RequestError>(),
                        Some(
                            teloxide::RequestError::Api(_)
                                | teloxide::RequestError::RetryAfter(_)
                                | teloxide::RequestError::MigrateToChatId(_)
                        )
                    ) {
                        job.kick_started = false;
                        runtime
                            .save_captcha(&job, Utc::now().timestamp() + 30)
                            .await?;
                    }
                    return Err(err);
                }
                job.state = "unkick".into();
                runtime.save_captcha(&job, 0).await?;
            }
            "unkick" => {
                let member =
                    telegram(runtime, async { bot.get_chat_member(chat, user).await }).await?;
                if owns_kick(&member.kind, job.kick_until)
                    && !independent_ban(runtime, &job).await?
                {
                    telegram(runtime, async {
                        bot.unban_chat_member(chat, user).only_if_banned(true).await
                    })
                    .await?;
                }
                job.state = "cleanup".into();
                runtime.save_captcha(&job, 0).await?;
            }
            "cleanup" => {
                if let Some(message_id) = job.question {
                    let result = telegram(runtime, async {
                        bot.delete_message(chat, MessageId(message_id)).await
                    })
                    .await;
                    if let Err(err) = result {
                        if !err
                            .to_string()
                            .to_ascii_lowercase()
                            .contains("message to delete not found")
                        {
                            return Err(err);
                        }
                    }
                }
                runtime.delete_captcha(job.chat_id, job.user_id).await?;
                return Ok(());
            }
            _ => anyhow::bail!("unknown CAPTCHA state"),
        }
    }
    Ok(())
}

async fn attempt_captcha(bot: &Bot, runtime: &Runtime, job: Challenge) {
    if let Err(err) = run_captcha(bot, runtime, job.clone()).await {
        log::warn!("captcha {} / {}: {err}", job.chat_id, job.user_id);
        let error = err.to_string().chars().take(2000).collect::<String>();
        if let Err(err) = runtime.with_conn(move |conn| {
            // A transition may have made the job due immediately before an API failure.
            conn.execute("UPDATE captcha_jobs SET last_error=?3,next_attempt_at=MAX(next_attempt_at,?4) WHERE chat_id=?1 AND user_id=?2",
                params![job.chat_id,job.user_id,error,Utc::now().timestamp()+5])?;
            Ok(())
        }).await { log::warn!("could not save CAPTCHA retry: {err}"); }
    }
}

pub(super) async fn start_captcha_challenge(
    bot: &Bot,
    runtime: &Arc<Runtime>,
    message: &Message,
    user: &teloxide::types::User,
) {
    if !message.chat.is_supergroup() || runtime.config.test_group_id == Some(message.chat.id.0) {
        return;
    }
    let user_id = user.id.0 as i64;
    let guard = Arc::new(runtime.user_action_guard(user_id).await);
    let result: Result<()> = async {
        let controls = runtime.operations_controls().await?;
        if controls.automatic_new_paused || controls.automatic_pending_paused {
            return Ok(());
        }
        let mut prior_restrict_until = None;
        if let Some(old) = runtime.captcha_job(message.chat.id.0, user_id).await? {
            if old.join_message_id == message.id.0 {
                return Ok(());
            }
            prior_restrict_until = Some(old.restrict_until);
            // A genuine new join replaces the old timeout, under the same lock.
            if let Some(id) = old.question {
                let _ = telegram(runtime, async {
                    bot.delete_message(message.chat.id, MessageId(id)).await
                })
                .await;
            }
        }
        let entropy = Uuid::new_v4();
        let now = Utc::now().timestamp();
        let job = Challenge {
            control_epoch: 0,
            chat_id: message.chat.id.0,
            user_id,
            join_message_id: message.id.0,
            name: short_user(user),
            a: entropy.as_bytes()[0] % 8 + 1,
            b: entropy.as_bytes()[1] % 8 + 1,
            deadline: now + ANSWER_SECONDS,
            restrict_until: now + ANSWER_SECONDS + 60,
            prior_restrict_until,
            question: None,
            state: "prepare".into(),
            kick_until: None,
            kick_started: false,
        };
        let Some(job) = runtime.queue_captcha(job, guard.clone()).await? else {
            return Ok(());
        };
        if runtime.claim_captcha(job.chat_id, job.user_id).await? {
            attempt_captcha(bot, runtime, job).await;
        }
        Ok(())
    }
    .await;
    if let Err(err) = result {
        log::warn!("could not start CAPTCHA: {err}");
    }
}

pub(super) async fn check_captcha_and_act(
    bot: &Bot,
    runtime: &Arc<Runtime>,
    message: &Message,
) -> bool {
    let Some(user) = message.from.as_ref() else {
        return false;
    };
    let _guard = runtime.user_action_guard(user.id.0 as i64).await;
    let mut job = match runtime
        .captcha_job(message.chat.id.0, user.id.0 as i64)
        .await
    {
        Ok(Some(job)) => job,
        Ok(None) => return false,
        Err(err) => {
            log::warn!("could not read CAPTCHA: {err}");
            return true;
        }
    };
    if matches!(job.state.as_str(), "cleanup" | "unkick") {
        return false;
    }
    let cancelled = match runtime.captcha_control_epoch().await {
        Ok(epoch) => epoch != job.control_epoch,
        Err(err) => {
            log::warn!("could not read CAPTCHA control: {err}");
            return true;
        }
    };
    if cancelled
        && !job.kick_started
        && matches!(job.state.as_str(), "prepare" | "waiting" | "kick")
    {
        job.state = "release".into();
        if let Err(err) = runtime.save_captcha(&job, 0).await {
            log::warn!("could not release paused CAPTCHA: {err}");
            return true;
        }
    }
    if matches!(job.state.as_str(), "prepare" | "waiting")
        && Utc::now().timestamp() < job.deadline
        && message.text().unwrap_or("").trim() == (job.a + job.b).to_string()
    {
        job.state = "release".into();
        if let Err(err) = runtime.save_captcha(&job, 0).await {
            log::warn!("could not save CAPTCHA answer: {err}");
            return true;
        }
    }
    if !cancelled {
        let _ = telegram(runtime, async {
            bot.delete_message(message.chat.id, message.id).await
        })
        .await;
    }
    if runtime
        .claim_captcha(job.chat_id, job.user_id)
        .await
        .unwrap_or(false)
    {
        attempt_captcha(bot, runtime, job).await;
    }
    true
}

pub(super) async fn retry_captchas(bot: &Bot, runtime: &Runtime) -> Result<usize> {
    let keys=runtime.with_conn(|conn| {
        let mut stmt=conn.prepare("SELECT chat_id,user_id FROM captcha_jobs WHERE next_attempt_at<=?1 ORDER BY next_attempt_at LIMIT 20")?;
        let rows=stmt.query_map(params![Utc::now().timestamp()],|r| Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?)))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }).await?;
    let mut attempted = 0;
    for (chat_id, user_id) in keys {
        let _guard = runtime.user_action_guard(user_id).await;
        if runtime.claim_captcha(chat_id, user_id).await? {
            if let Some(job) = runtime.captcha_job(chat_id, user_id).await? {
                attempted += 1;
                attempt_captcha(bot, runtime, job).await;
            }
        }
    }
    Ok(attempted)
}

pub(super) fn spawn_captcha_worker(bot: Bot, runtime: Arc<Runtime>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            if let Err(err) = retry_captchas(&bot, &runtime).await {
                log::warn!("CAPTCHA queue: {err}");
            }
            sleep(Duration::from_secs(5)).await;
        }
    })
}
