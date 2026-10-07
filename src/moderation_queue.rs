use super::*;
use rusqlite::OptionalExtension;

impl Runtime {
    pub(super) fn migrate_v23_to_v24(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS ban_followups (
            case_id TEXT PRIMARY KEY,
            training_mode TEXT NOT NULL CHECK(training_mode IN ('review','direct','report')),
            training_done INTEGER NOT NULL DEFAULT 0,
            review_sent INTEGER NOT NULL DEFAULT 0,
            audit_id INTEGER,
            audit_done INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS moderation_requests (
            chat_id INTEGER NOT NULL,message_id INTEGER NOT NULL,case_id TEXT NOT NULL,
            PRIMARY KEY(chat_id,message_id)
        );
        CREATE TABLE IF NOT EXISTS review_updates (
            case_id TEXT PRIMARY KEY,kind TEXT NOT NULL,decision TEXT NOT NULL,actor_id INTEGER,
            chat_id INTEGER NOT NULL,message_id INTEGER NOT NULL,
            confirmation_chat_id INTEGER,confirmation_message_id INTEGER,
            note TEXT NOT NULL DEFAULT '',review_status TEXT NOT NULL DEFAULT '',
            confirmation_status TEXT NOT NULL DEFAULT '',attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt_at INTEGER NOT NULL DEFAULT 0,last_error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_review_update_due ON review_updates(next_attempt_at);
        PRAGMA user_version=24;",
        )?;
        tx.commit()?;
        Ok(())
    }

    pub(super) async fn queue_manual_ban(
        &self,
        case: CaseRecord,
        force: bool,
        command_id: i32,
    ) -> Result<String> {
        anyhow::ensure!(
            case.action == ActionKind::SpamBan,
            "invalid manual ban action"
        );
        let review = self.review_guard(&case.id).await;
        let user = self.user_action_guard(case.target_user_id).await;
        self.with_conn(move |conn| {
            let _guards=(review,user);
            let tx=conn.transaction()?;
            let existing=tx.query_row("SELECT case_id FROM moderation_requests WHERE chat_id=?1 AND message_id=?2",
                params![case.chat_id,command_id],|r|r.get::<_,String>(0)).optional()?;
            if let Some(id)=existing { return Ok(id); }
            anyhow::ensure!(origin_retry::insert_origin_case(&tx,&case,"<b>已執行管理操作</b>")?,"case already exists");
            let undo=serde_json::to_string(&UndoData::Case {case_id:case.id.clone(),kind:CaseKind::Ban})?;
            tx.execute("INSERT INTO maintainer_actions(actor_id,actor_name,chat_id,command,summary,undo_data,created_at)
                VALUES (?1,?2,?3,?4,?5,?6,?7)",
                params![case.actor_user_id,case.actor_name,case.chat_id,if force {"/sb -f"} else {"/sb"},
                    format!("封禁請求 對象={}",case.target_user_id),undo,Utc::now().to_rfc3339()])?;
            let audit_id=tx.last_insert_rowid();
            tx.execute("INSERT INTO ban_followups(case_id,training_mode,audit_id) VALUES (?1,?2,?3)",
                params![case.id,if force {"direct"} else {"review"},audit_id])?;
            tx.execute("INSERT INTO moderation_requests(chat_id,message_id,case_id) VALUES (?1,?2,?3)",
                params![case.chat_id,command_id,case.id])?;
            tx.commit()?;
            Ok(case.id)
        }).await
    }

    pub(super) async fn decide_report(
        &self,
        case: &CaseRecord,
        decision: &str,
        actor: (i64, String),
        location: (i64, i32),
        guard: tokio::sync::OwnedMutexGuard<()>,
    ) -> Result<bool> {
        anyhow::ensure!(
            matches!(decision, "approve" | "reject"),
            "invalid report decision"
        );
        let strike_reporter = if let Some(id) = case.actor_user_id {
            !self.is_maintainer(id).await
        } else {
            false
        };
        let case_id = case.id.clone();
        let decision = decision.to_string();
        self.with_model_transaction(move |tx| {
            let _guard=guard;
            let pending: Option<(String,Option<i64>,bool)>=tx.query_row(
                "SELECT evidence_text,actor_user_id,source_message_id IS NULL FROM cases
                 WHERE id=?1 AND action='pending_report' AND status='pending_review'",params![case_id],
                |r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
            let Some((text,reporter,no_source))=pending else {return Ok((false,_guard));};
            let mut note=String::new();
            if decision == "approve" {
                tx.execute("UPDATE cases SET action='report_approved',status='ban_pending',actor_user_id=?2,actor_name=?3,log_message_id=NULL WHERE id=?1",
                    params![case_id,actor.0,actor.1])?;
                tx.execute("INSERT INTO origin_ban_jobs(case_id,header,delete_done) VALUES (?1,'<b>舉報已受理，對象已封禁</b>',?2)",
                    params![case_id,no_source])?;
                tx.execute("INSERT INTO ban_followups(case_id,training_mode,audit_done) VALUES (?1,'report',1)",params![case_id])?;
            } else {
                reliability::write_sample(tx,"ham",&text,Some(&case_id))?;
                if let Some(reporter)=reporter.filter(|_|strike_reporter) {
                    tx.execute("INSERT INTO report_offenses(user_id,rejected_count,last_rejected_at) VALUES (?1,1,?2)
                        ON CONFLICT(user_id) DO UPDATE SET rejected_count=rejected_count+1,last_rejected_at=excluded.last_rejected_at",
                        params![reporter,Utc::now().to_rfc3339()])?;
                    let count: i64=tx.query_row("SELECT rejected_count FROM report_offenses WHERE user_id=?1",[reporter],|r|r.get(0))?;
                    note=if count>=REPORT_STRIKE_LIMIT {
                        format!("\n<b>舉報者</b>: <code>{reporter}</code> 已累計 {count} 次被拒，已暫停使用 /spam")
                    } else {format!("\n<b>舉報者</b>: <code>{reporter}</code> 已累計 {count}/{REPORT_STRIKE_LIMIT} 次被拒")};
                }
                tx.execute("UPDATE cases SET action='report_rejected',status='rejected_and_cleaned',actor_user_id=?2,actor_name=?3 WHERE id=?1",
                    params![case_id,actor.0,actor.1])?;
            }
            tx.execute("INSERT INTO review_updates(case_id,kind,decision,chat_id,message_id,confirmation_chat_id,confirmation_message_id,note)
                SELECT ?1,'report',?2,?3,?4,(SELECT chat_id FROM report_confirmations WHERE case_id=?1),
                (SELECT message_id FROM report_confirmations WHERE case_id=?1),?5",params![case_id,decision,location.0,location.1,note])?;
            tx.execute("DELETE FROM report_confirmations WHERE case_id=?1",params![case_id])?;
            Ok((true,_guard))
        }).await.map(|(changed,_)|changed)
    }
}

pub(super) async fn deliver_review_updates(
    bot: &Bot,
    runtime: &Runtime,
    filter: Option<&str>,
) -> Result<()> {
    let filter = filter.map(str::to_string);
    let ids=runtime.with_conn(move |conn| {
        let mut stmt=conn.prepare("SELECT u.case_id FROM review_updates u JOIN cases c ON c.id=u.case_id
            LEFT JOIN origin_ban_jobs j ON j.case_id=c.id WHERE (?1 IS NULL OR u.case_id=?1) AND u.next_attempt_at<=?2
            AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?2
            AND (u.review_status != c.status||':'||COALESCE(j.state,'') OR u.confirmation_status != c.status||':'||COALESCE(j.state,''))
            ORDER BY u.next_attempt_at,u.case_id LIMIT 20")?;
        let rows=stmt.query_map(params![filter,Utc::now().timestamp()],|r|r.get::<_,String>(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }).await?;
    for id in ids {
        if let Err(err) = deliver_review_update(bot, runtime, &id).await {
            log::warn!(
                "review notification {id}: {}",
                notices::diagnostic(&runtime.config, &err.to_string())
            );
        }
    }
    Ok(())
}

async fn deliver_review_update(bot: &Bot, runtime: &Runtime, id: &str) -> Result<()> {
    let guard = Arc::new(runtime.review_guard(id).await);
    let Some(case) = runtime.load_case(id).await? else {
        return Ok(());
    };
    let case_id = id.to_string();
    let claim_guard = guard.clone();
    let item=runtime.with_conn(move |conn| {
        let _guard=claim_guard;
        let tx=conn.transaction()?;
        let now=Utc::now().timestamp();
        let row=tx.query_row("SELECT u.chat_id,u.message_id,u.confirmation_chat_id,u.confirmation_message_id,u.note,
            u.review_status,u.confirmation_status,u.attempts,COALESCE(j.state,''),u.kind,u.decision,u.actor_id FROM review_updates u
            LEFT JOIN origin_ban_jobs j ON j.case_id=u.case_id WHERE u.case_id=?1 AND u.next_attempt_at<=?2
            AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?2",params![case_id,now],
            |r|Ok((r.get::<_,i64>(0)?,r.get::<_,i32>(1)?,r.get::<_,Option<i64>>(2)?,r.get::<_,Option<i32>>(3)?,
                r.get::<_,String>(4)?,r.get::<_,String>(5)?,r.get::<_,String>(6)?,r.get::<_,u32>(7)?,r.get::<_,String>(8)?,r.get::<_,String>(9)?,r.get::<_,String>(10)?,r.get::<_,Option<i64>>(11)?))).optional()?;
        if let Some(ref row)=row {
            let attempt=row.7.saturating_add(1).min(31);
            let delay=(60_i64<<attempt.saturating_sub(1).min(6)).min(3600);
            tx.execute("UPDATE review_updates SET attempts=?2,next_attempt_at=?3 WHERE case_id=?1",params![case_id,attempt,now+delay])?;
        }
        tx.commit()?;
        Ok(row)
    }).await?;
    let Some((
        chat,
        msg,
        confirmation_chat,
        confirmation_msg,
        note,
        review_status,
        confirmation_status,
        _,
        state,
        kind,
        decision,
        actor,
    )) = item
    else {
        return Ok(());
    };
    let key = format!("{}:{state}", case.status);
    let (title, confirmation) = if matches!(case.status.as_str(), "reversed" | "reversal_pending") {
        ("處理已撤銷", "此舉報的處理已撤銷。")
    } else if kind == "train" && decision == "approve" {
        ("已訓練並加入跨群黑名單", "")
    } else if kind == "train" {
        ("已拒絕訓練，本群封禁保留", "")
    } else if case.action == ActionKind::ReportRejected {
        ("已拒絕舉報", "此舉報未被受理。")
    } else if state == "cancelled" {
        (
            "已受理，操作已取消",
            "舉報已受理；權限或狀態已變更，封禁操作已取消。",
        )
    } else if matches!(case.status.as_str(), "ban_pending" | "ban_failed") {
        (
            "已受理，封禁待處理",
            "舉報已受理，封禁尚未完成，系統會自動重試。",
        )
    } else {
        ("已受理並封禁", "舉報已受理，對象已被封禁。")
    };
    for (stage, chat, msg, text, status) in [
        (
            "review_status",
            Some(chat),
            Some(msg),
            notices::review_card(&case, title, &note, actor.or(case.actor_user_id)),
            review_status,
        ),
        (
            "confirmation_status",
            confirmation_chat,
            confirmation_msg,
            confirmation.to_string(),
            confirmation_status,
        ),
    ] {
        if status == key {
            continue;
        }
        if let (Some(chat), Some(msg)) = (chat, msg) {
            let request = bot
                .edit_message_text(ChatId(chat), MessageId(msg), text)
                .parse_mode(ParseMode::Html)
                .reply_markup(InlineKeyboardMarkup::new(
                    Vec::<Vec<InlineKeyboardButton>>::new(),
                ));
            let result = tokio::time::timeout(Duration::from_secs(30), async { request.await })
                .await
                .context("Telegram request timed out")
                .and_then(|r| r.map_err(anyhow::Error::from));
            if let Err(err) = result {
                if !err.to_string().contains("message is not modified")
                    && !err.to_string().contains("message to edit not found")
                {
                    if let Some(teloxide::RequestError::RetryAfter(delay)) =
                        err.downcast_ref::<teloxide::RequestError>()
                    {
                        runtime.delay_telegram_queue(delay.seconds()).await?;
                    }
                    let id = id.to_string();
                    let error = notices::diagnostic(&runtime.config, &err.to_string());
                    let error = error.chars().take(1000).collect::<String>();
                    let guard = guard.clone();
                    runtime
                        .with_conn(move |conn| {
                            let _guard = guard;
                            conn.execute(
                                "UPDATE review_updates SET last_error=?2 WHERE case_id=?1",
                                params![id, error],
                            )?;
                            Ok(())
                        })
                        .await?;
                    return Err(err);
                }
            }
        }
        let id = id.to_string();
        let key = key.clone();
        let guard = guard.clone();
        runtime
            .with_conn(move |conn| {
                let _guard = guard;
                conn.execute(
                    &format!(
                        "UPDATE review_updates SET {stage}=?2,last_error=NULL WHERE case_id=?1"
                    ),
                    params![id, key],
                )?;
                Ok(())
            })
            .await?;
    }
    Ok(())
}
