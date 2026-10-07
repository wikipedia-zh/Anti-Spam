use super::*;
use rusqlite::OptionalExtension;
use teloxide::types::{ChatMemberKind, ChatPermissions, UntilDate};

#[derive(Clone, Serialize, Deserialize)]
struct Job {
    step: String,
    until: Option<i64>,
    kick_until: Option<i64>,
    uncertain: bool,
    header: String,
    audit_id: Option<i64>,
    audit_sent: bool,
    notice_sent: bool,
    release_mode: String,
    release_fingerprint: Option<String>,
}

struct Guards {
    _case: tokio::sync::OwnedMutexGuard<()>,
    _user: tokio::sync::OwnedMutexGuard<()>,
}

impl Runtime {
    pub(super) fn migrate_v24_to_v25(conn: &mut Connection) -> Result<()> {
        conn.execute_batch(
            "BEGIN;
            CREATE TABLE IF NOT EXISTS restriction_jobs (
                case_id TEXT PRIMARY KEY, state TEXT NOT NULL DEFAULT 'pending',
                payload TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at INTEGER NOT NULL DEFAULT 0,last_error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_restriction_due ON restriction_jobs(state,next_attempt_at);
            PRAGMA user_version=25;
            COMMIT;",
        )?;
        Ok(())
    }

    async fn restriction_guards(&self, case: &CaseRecord) -> Arc<Guards> {
        Arc::new(Guards {
            _case: self.review_guard(&case.id).await,
            _user: self.user_action_guard(case.target_user_id).await,
        })
    }

    pub(super) async fn queue_restriction(
        &self,
        case: CaseRecord,
        message_id: i32,
        until: Option<i64>,
        header: &str,
    ) -> Result<String> {
        anyhow::ensure!(
            matches!(
                case.action,
                ActionKind::Mute
                    | ActionKind::Kick
                    | ActionKind::FloodMute
                    | ActionKind::CmdCleanMute
            ),
            "unsupported restriction"
        );
        let guards = self.restriction_guards(&case).await;
        let header = header.to_string();
        self.with_conn(move |conn| {
            let _guards=guards;
            let tx=conn.transaction()?;
            if let Some(id)=tx.query_row("SELECT case_id FROM moderation_requests WHERE chat_id=?1 AND message_id=?2",
                params![case.chat_id,message_id],|r|r.get::<_,String>(0)).optional()? {return Ok(id);}
            tx.execute("INSERT INTO cases(id,action,chat_id,target_user_id,target_name,actor_user_id,actor_name,source_message_id,evidence_text,model_score,matched_rule_id,matched_rule_pattern,status,log_message_id,created_at)
                VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,'action_pending',NULL,?13)",
                params![case.id,case.action.as_str(),case.chat_id,case.target_user_id,case.target_name,case.actor_user_id,case.actor_name,
                    case.source_message_id,case.evidence_text,case.model_score,case.matched_rule_id,case.matched_rule_pattern,case.created_at.to_rfc3339()])?;
            let mut job=Job {step:"apply".into(),until,kick_until:None,uncertain:false,header,audit_id:None,audit_sent:false,notice_sent:false,release_mode:"known".into(),release_fingerprint:None};
            if let Some(actor)=case.actor_user_id {
                let undo=if case.action==ActionKind::Kick {UndoData::NotRevertible} else {UndoData::Case{case_id:case.id.clone(),kind:CaseKind::Mute}};
                tx.execute("INSERT INTO maintainer_actions(actor_id,actor_name,chat_id,command,summary,undo_data,created_at) VALUES (?1,?2,?3,?4,?5,?6,?7)",
                    params![actor,case.actor_name,case.chat_id,if case.action==ActionKind::Kick {"/kick"} else {"/mute"},
                    format!("管理請求 對象={}",case.target_user_id),serde_json::to_string(&undo)?,Utc::now().to_rfc3339()])?;
                job.audit_id=Some(tx.last_insert_rowid());
            }
            tx.execute("INSERT INTO restriction_jobs(case_id,payload) VALUES (?1,?2)",params![case.id,serde_json::to_string(&job)?])?;
            tx.execute("INSERT INTO moderation_requests(chat_id,message_id,case_id) VALUES (?1,?2,?3)",params![case.chat_id,message_id,case.id])?;
            tx.commit()?;
            Ok(case.id)
        }).await
    }

    async fn save_restriction(
        &self,
        case: &CaseRecord,
        job: &Job,
        state: &str,
        guards: Arc<Guards>,
    ) -> Result<()> {
        let case = case.clone();
        let payload = serde_json::to_string(job)?;
        let state = state.to_string();
        self.with_conn(move |conn| {
            let _guards=guards;
            let tx=conn.transaction()?;
            tx.execute("UPDATE restriction_jobs SET payload=?2,state=?3,last_error=NULL WHERE case_id=?1",params![case.id,payload,state])?;
            tx.execute("UPDATE cases SET status=?2,action=?3,log_message_id=?4,actor_user_id=?5,actor_name=?6 WHERE id=?1",
                params![case.id,case.status,case.action.as_str(),case.log_message_id,case.actor_user_id,case.actor_name])?;
            tx.commit()?;
            Ok(())
        }).await
    }
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

fn owns_kick(member: &ChatMemberKind, until: Option<i64>) -> bool {
    matches!((member,until),(ChatMemberKind::Banned(b),Some(t)) if b.until_date==UntilDate::Date(DateTime::from_timestamp(t,0).unwrap()))
}

fn owns_mute(member: &ChatMemberKind, until: Option<i64>) -> bool {
    matches!(member,ChatMemberKind::Restricted(r) if !r.can_send_messages && r.until_date==until.map(|t|UntilDate::Date(DateTime::from_timestamp(t,0).unwrap())).unwrap_or(UntilDate::Forever))
}

async fn run(
    bot: &Bot,
    runtime: &Runtime,
    case: &mut CaseRecord,
    job: &mut Job,
    guards: Arc<Guards>,
) -> Result<()> {
    let chat = ChatId(case.chat_id);
    let user = UserId(case.target_user_id as u64);
    if job.step == "apply" {
        let member = telegram(runtime, async { bot.get_chat_member(chat, user).await }).await?;
        if job.uncertain {
            // An interrupted kick must never remove somebody who has rejoined.
            if case.action == ActionKind::Kick {
                job.step = if owns_kick(&member.kind, job.kick_until) {
                    "unkick"
                } else {
                    "notice"
                }
                .into();
                case.status = if job.step == "unkick" {
                    "kick_release_pending"
                } else {
                    "action_unconfirmed"
                }
                .into();
            } else if owns_mute(&member.kind, job.until) {
                job.step = "notice".into();
                case.status = "done".into();
            } else {
                case.status = "action_unconfirmed".into();
                job.step = "notice".into();
            }
            runtime
                .save_restriction(case, job, "pending", guards.clone())
                .await?;
        } else {
            let mut allowed = !member.kind.is_privileged()
                && !runtime.is_maintainer(case.target_user_id).await
                && !is_platform_pseudo_user(case.target_user_id)
                && !runtime.is_group_banned(case.chat_id).await;
            if let Some(actor) = case.actor_user_id {
                allowed &= telegram(runtime, async {
                    bot.get_chat_member(chat, UserId(actor as u64)).await
                })
                .await?
                .kind
                .is_privileged();
            }
            if case.action == ActionKind::FloodMute {
                allowed &= runtime.get_group_modules(case.chat_id).await?.flood_control
                    && !runtime.is_global_whitelisted(case.target_user_id).await?
                    && !runtime
                        .is_group_whitelisted(case.chat_id, case.target_user_id)
                        .await?;
            }
            if case.action == ActionKind::CmdCleanMute {
                allowed &= runtime.get_group_modules(case.chat_id).await?.cmd_clean;
            }
            if job
                .until
                .is_some_and(|until| until < Utc::now().timestamp() + 35)
            {
                allowed = false;
            }
            // A finite mute/kick cannot replace an independent permanent ban or mute.
            if runtime
                .has_other_ban_in_chat(&case.id, case.chat_id, case.target_user_id)
                .await?
                || matches!(member.kind, ChatMemberKind::Banned(_))
            {
                allowed = false;
            }
            if job.until.is_some()
                && matches!(member.kind, ChatMemberKind::Restricted(_))
                && !owns_mute(&member.kind, job.until)
            {
                allowed = false;
            }
            if !allowed {
                case.status = "action_cancelled".into();
                return runtime
                    .save_restriction(case, job, "cancelled", guards)
                    .await;
            }
            job.uncertain = true;
            if case.action == ActionKind::Kick {
                job.kick_until = Some(Utc::now().timestamp() + 90);
            }
            runtime
                .save_restriction(case, job, "pending", guards.clone())
                .await?;
            let result = if case.action == ActionKind::Kick {
                telegram(runtime, async {
                    bot.ban_chat_member(chat, user)
                        .until_date(DateTime::from_timestamp(job.kick_until.unwrap(), 0).unwrap())
                        .await
                })
                .await
            } else {
                telegram(runtime, async {
                    let request = bot.restrict_chat_member(chat, user, ChatPermissions::empty());
                    match job.until {
                        Some(t) => {
                            request
                                .until_date(DateTime::from_timestamp(t, 0).unwrap())
                                .await
                        }
                        None => request.await,
                    }
                })
                .await
            };
            if let Err(err) = result {
                if matches!(
                    err.downcast_ref::<teloxide::RequestError>(),
                    Some(
                        teloxide::RequestError::Api(_)
                            | teloxide::RequestError::RetryAfter(_)
                            | teloxide::RequestError::MigrateToChatId(_)
                    )
                ) {
                    job.uncertain = false;
                }
                case.status = "action_failed".into();
                runtime
                    .save_restriction(case, job, "pending", guards)
                    .await?;
                return Err(err);
            }
            job.uncertain = false;
            job.step = if case.action == ActionKind::Kick {
                "unkick"
            } else {
                "notice"
            }
            .into();
            case.status = if case.action == ActionKind::Kick {
                "kick_release_pending"
            } else {
                "done"
            }
            .into();
            runtime
                .save_restriction(case, job, "pending", guards.clone())
                .await?;
        }
    }
    if job.step == "unkick" {
        let member = telegram(runtime, async { bot.get_chat_member(chat, user).await }).await?;
        if owns_kick(&member.kind, job.kick_until)
            && !runtime
                .has_other_ban_in_chat(&case.id, case.chat_id, case.target_user_id)
                .await?
        {
            telegram(runtime, async {
                bot.unban_chat_member(chat, user).only_if_banned(true).await
            })
            .await?;
        }
        job.step = "notice".into();
        case.status = "done".into();
        runtime
            .save_restriction(case, job, "pending", guards.clone())
            .await?;
    }
    if job.step == "unmute" {
        let id = case.id.clone();
        let chat_id = case.chat_id;
        let target = case.target_user_id;
        let other=runtime.with_conn(move |conn| Ok(conn.query_row("SELECT EXISTS(SELECT 1 FROM cases c LEFT JOIN restriction_jobs j ON j.case_id=c.id
            WHERE c.id!=?1 AND c.chat_id=?2 AND c.target_user_id=?3 AND c.action IN ('mute','flood_mute','cmd_clean_mute')
            AND c.status NOT IN ('reversed','reversal_pending','locally_unmuted','action_cancelled','action_failed','action_pending','action_unconfirmed')
            AND (j.case_id IS NULL OR json_extract(j.payload,'$.until') IS NULL OR json_extract(j.payload,'$.until')>?4))",
            params![id,chat_id,target,Utc::now().timestamp()],|r|r.get::<_,bool>(0))?)).await?;
        let member = telegram(runtime, async { bot.get_chat_member(chat, user).await }).await?;
        let owned = job.release_mode == "all"
            || (job.release_mode == "known" && owns_mute(&member.kind, job.until));
        let fingerprint = serde_json::to_string(&member.kind)?;
        let unchanged = job
            .release_fingerprint
            .as_ref()
            .is_none_or(|old| old == &fingerprint);
        if job.release_fingerprint.is_none() {
            job.release_fingerprint = Some(fingerprint);
            runtime
                .save_restriction(case, job, "pending", guards.clone())
                .await?;
        }
        if !other && owned && unchanged && matches!(member.kind, ChatMemberKind::Restricted(_)) {
            telegram(runtime, async {
                bot.restrict_chat_member(chat, user, ChatPermissions::all())
                    .await
            })
            .await?;
        }
        job.step = "notice".into();
        case.action = ActionKind::Unmuted;
        case.status = "reversed".into();
        if other
            || !unchanged
            || (job.release_mode == "known"
                && !owned
                && matches!(member.kind, ChatMemberKind::Restricted(_)))
        {
            job.header = "<b>已撤銷此禁言；其他限制仍保留</b>".into();
        }
        runtime
            .save_restriction(case, job, "pending", guards.clone())
            .await?;
    }
    if job.step == "notice" {
        if !job.notice_sent {
            let result = if let Some(id) = case.log_message_id {
                telegram(runtime, async {
                    bot.edit_message_text(
                        ChatId(runtime.config.log_channel_id),
                        MessageId(id),
                        notices::action_log(case),
                    )
                    .parse_mode(ParseMode::Html)
                    .await
                })
                .await
                .map(|_| id)
            } else {
                telegram(runtime, log_action(bot, runtime, case)).await
            };
            match result {
                Ok(id) => case.log_message_id = Some(id),
                Err(err) if err.to_string().contains("message is not modified") => {}
                Err(err) if err.to_string().contains("message to edit not found") => {
                    case.log_message_id = None;
                    runtime
                        .save_restriction(case, job, "pending", guards.clone())
                        .await?;
                    return Err(err);
                }
                Err(err) => return Err(err),
            }
            runtime
                .save_restriction(case, job, "pending", guards.clone())
                .await?;
            let header = if case.status == "action_unconfirmed" {
                "<b>操作結果未確認，請管理員檢查</b>"
            } else {
                &job.header
            };
            tokio::time::timeout(
                Duration::from_secs(30),
                notify_group(
                    bot,
                    runtime,
                    case,
                    case.log_message_id.context("missing log")?,
                    header,
                ),
            )
            .await
            .context("Telegram request timed out")??;
            job.notice_sent = true;
            runtime
                .save_restriction(case, job, "pending", guards.clone())
                .await?;
        }
        if !job.audit_sent {
            if let (Some(id), Some(chat)) = (job.audit_id, runtime.audit_log_chat().await) {
                let hint = if case.action == ActionKind::Kick {
                    "（無法復原）".to_string()
                } else {
                    format!("復原：<code>/revert {id}</code>")
                };
                let text=format!("<b>維護操作 #{id}</b>\n{}\n<b>操作者</b>: <code>{}</code>\n<b>群組</b>: <code>{}</code>\n<b>對象</b>: <code>{}</code>\n{hint}",chinese_case_action(case),case.actor_user_id.unwrap_or(0),case.chat_id,case.target_user_id);
                telegram(runtime, async {
                    bot.send_message(ChatId(chat), text)
                        .parse_mode(ParseMode::Html)
                        .await
                })
                .await?;
            }
            job.audit_sent = true;
        }
        runtime.save_restriction(case, job, "done", guards).await?;
    }
    Ok(())
}

pub(super) async fn attempt(bot: &Bot, runtime: &Runtime, case: CaseRecord) -> Result<()> {
    let guards = runtime.restriction_guards(&case).await;
    let Some(mut case) = runtime.load_case(&case.id).await? else {
        return Ok(());
    };
    let id = case.id.clone();
    let claim_guard = guards.clone();
    let payload=runtime.with_conn(move |conn| {
        let _guard=claim_guard;
        let tx=conn.transaction()?;
        let row=tx.query_row("SELECT payload,attempts FROM restriction_jobs WHERE case_id=?1 AND state='pending' AND next_attempt_at<=?2
            AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?2",params![id,Utc::now().timestamp()],|r|Ok((r.get::<_,String>(0)?,r.get::<_,u32>(1)?))).optional()?;
        if let Some((_,attempts))=row.as_ref() {
            let n=attempts.saturating_add(1).min(31);
            tx.execute("UPDATE restriction_jobs SET attempts=?2,next_attempt_at=?3 WHERE case_id=?1",params![id,n,Utc::now().timestamp()+(60_i64<<n.saturating_sub(1).min(6)).min(3600)])?;
        }
        tx.commit()?;Ok(row.map(|r|r.0))
    }).await?;
    let Some(payload) = payload else {
        return Ok(());
    };
    let mut job: Job = serde_json::from_str(&payload)?;
    if case.status == "reversed" && job.step != "notice" {
        return runtime
            .save_restriction(&case, &job, "cancelled", guards)
            .await;
    }
    if let Err(err) = run(bot, runtime, &mut case, &mut job, guards.clone()).await {
        if let Some(teloxide::RequestError::RetryAfter(delay)) =
            err.downcast_ref::<teloxide::RequestError>()
        {
            runtime.delay_telegram_queue(delay.seconds()).await?;
        }
        let error = notices::diagnostic(&runtime.config, &err.to_string())
            .chars()
            .take(1000)
            .collect::<String>();
        log::warn!("restriction {}: {error}", case.id);
        let id = case.id.clone();
        runtime
            .with_conn(move |conn| {
                let _guard = guards;
                conn.execute(
                    "UPDATE restriction_jobs SET last_error=?2 WHERE case_id=?1",
                    params![id, error],
                )?;
                Ok(())
            })
            .await?;
        return Err(err);
    }
    Ok(())
}

pub(super) async fn retry(bot: &Bot, runtime: &Runtime) -> Result<()> {
    let ids=runtime.with_conn(|conn| {
        let mut stmt=conn.prepare("SELECT case_id FROM restriction_jobs WHERE state='pending' AND next_attempt_at<=?1
            AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?1 ORDER BY next_attempt_at,case_id LIMIT 20")?;
        let rows=stmt.query_map([Utc::now().timestamp()],|r|r.get::<_,String>(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }).await?;
    for id in ids {
        if let Some(case) = runtime.load_case(&id).await? {
            let _ = attempt(bot, runtime, case).await;
        }
    }
    Ok(())
}

pub(super) async fn reverse(
    bot: &Bot,
    runtime: &Runtime,
    case: CaseRecord,
    actor_id: i64,
    actor_name: &str,
) -> Result<String, String> {
    async fn queue(
        runtime: &Runtime,
        case: &CaseRecord,
        actor_id: i64,
        actor_name: &str,
    ) -> Result<bool> {
        let guards = runtime.restriction_guards(case).await;
        let id = case.id.clone();
        let name = actor_name.to_string();
        runtime.with_conn(move |conn| {
            let _guards=guards;
            let tx=conn.transaction()?;
            let (action,status):(String,String)=tx.query_row("SELECT action,status FROM cases WHERE id=?1",params![id],|r|Ok((r.get(0)?,r.get(1)?)))?;
            if status=="reversed" {return Ok(false);}
            anyhow::ensure!(matches!(action.as_str(),"mute"|"flood_mute"|"cmd_clean_mute"),"此案例不是禁言");
            if status!="reversal_pending" {
                let old=tx.query_row("SELECT payload FROM restriction_jobs WHERE case_id=?1",params![id],|r|r.get::<_,String>(0)).optional()?;
                let old=old.map(|s|serde_json::from_str::<Job>(&s)).transpose()?;
                let release_mode=match &old {
                    None=>"all",
                    Some(job) if job.uncertain || status=="done"=>"known",
                    _=>"none",
                }.to_string();
                let job=Job{step:"unmute".into(),until:old.and_then(|j|j.until),kick_until:None,uncertain:false,header:"<b>已解除禁言</b>".into(),audit_id:None,audit_sent:true,notice_sent:false,release_mode,release_fingerprint:None};
                tx.execute("INSERT INTO restriction_jobs(case_id,payload) VALUES (?1,?2) ON CONFLICT(case_id) DO UPDATE SET state='pending',payload=excluded.payload,next_attempt_at=0,attempts=0,last_error=NULL",params![id,serde_json::to_string(&job)?])?;
                tx.execute("UPDATE cases SET status='reversal_pending',actor_user_id=?2,actor_name=?3 WHERE id=?1",params![id,actor_id,name])?;
            }
            tx.commit()?;Ok(true)
        }).await
    }
    if !queue(runtime, &case, actor_id, actor_name)
        .await
        .map_err(|e| e.to_string())?
    {
        return Ok("此案例已撤銷".into());
    }
    let _ = attempt(bot, runtime, case.clone()).await;
    let current = runtime
        .load_case(&case.id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("案例不存在")?;
    Ok(if current.status == "reversed" {
        format!(
            "已撤銷禁言案例 <code>{}</code>；其他有效限制仍保留。",
            case.id
        )
    } else {
        "解除禁言尚未完成，系統會重試。".into()
    })
}

pub(super) async fn release_all(
    bot: &Bot,
    runtime: &Runtime,
    chat: i64,
    user: i64,
    actor: i64,
    name: &str,
    command_id: i32,
) -> Result<String> {
    let guard = runtime.user_action_guard(user).await;
    let name = name.to_string();
    let id=runtime.with_conn(move |conn| {
        let _guard=guard;
        let tx=conn.transaction()?;
        if let Some(id)=tx.query_row("SELECT case_id FROM moderation_requests WHERE chat_id=?1 AND message_id=?2",params![chat,command_id],|r|r.get::<_,String>(0)).optional()? {return Ok(id);}
        let id=Uuid::new_v4().to_string();
        tx.execute("UPDATE restriction_jobs SET state='cancelled' WHERE case_id IN (SELECT id FROM cases WHERE chat_id=?1 AND target_user_id=?2 AND action IN ('mute','flood_mute','cmd_clean_mute'))",params![chat,user])?;
        tx.execute("UPDATE cases SET status='locally_unmuted' WHERE chat_id=?1 AND target_user_id=?2 AND action IN ('mute','flood_mute','cmd_clean_mute') AND status!='reversed'",params![chat,user])?;
        tx.execute("INSERT INTO cases(id,action,chat_id,target_user_id,target_name,actor_user_id,actor_name,evidence_text,status,created_at)
            VALUES (?1,'unmuted',?2,?3,?4,?5,?6,'','reversal_pending',?7)",params![id,chat,user,format!("User{user}"),actor,name,Utc::now().to_rfc3339()])?;
        let job=Job{step:"unmute".into(),until:None,kick_until:None,uncertain:false,header:"<b>已解除禁言</b>".into(),audit_id:None,audit_sent:true,notice_sent:false,release_mode:"all".into(),release_fingerprint:None};
        tx.execute("INSERT INTO restriction_jobs(case_id,payload) VALUES (?1,?2)",params![id,serde_json::to_string(&job)?])?;
        tx.execute("INSERT INTO moderation_requests(chat_id,message_id,case_id) VALUES (?1,?2,?3)",params![chat,command_id,id])?;
        tx.commit()?;Ok(id)
    }).await?;
    if let Some(case) = runtime.load_case(&id).await? {
        let _ = attempt(bot, runtime, case).await;
    }
    let done = runtime
        .load_case(&id)
        .await?
        .is_some_and(|c| c.status == "reversed");
    Ok(if done {
        "已在本群解除禁言。"
    } else {
        "解除禁言尚未完成，系統會重試。"
    }
    .to_string())
}
