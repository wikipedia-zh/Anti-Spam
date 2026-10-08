use super::*;
use rusqlite::OptionalExtension;
use serde_json::{json, Value};
use std::future::IntoFuture;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Read {
    pub chat_id: i64,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Patch {
    pub request_id: String,
    pub chat_id: i64,
    pub expected_revision: String,
    pub reason: String,
    pub block_rejoin: bool,
}
pub(super) enum Outcome {
    Ready(Value),
    Forbidden,
    Invalid,
    Conflict,
}

fn protected(conn: &Connection, config: &Config, chat: i64) -> Result<bool> {
    if chat >= 0
        || chat == config.log_channel_id
        || chat == config.report_channel_id
        || Some(chat) == config.test_group_id
    {
        return Ok(true);
    }
    Ok(conn.query_row("SELECT EXISTS(SELECT 1 FROM model_meta WHERE key IN ('project_chat_id','audit_log_chat_id','exchange_channel_id') AND CAST(value AS INTEGER)=?1)",[chat],|r|r.get(0))?)
}
fn latest(conn: &Connection, chat: i64) -> Result<Option<String>> {
    Ok(conn
        .query_row(
            "SELECT request_id FROM group_departures WHERE chat_id=?1 ORDER BY rowid DESC LIMIT 1",
            [chat],
            |r| r.get(0),
        )
        .optional()?)
}
fn job(conn: &Connection, id: &str) -> Result<Value> {
    Ok(conn.query_row("SELECT request_id,chat_id,title,reason,block_rejoin,state,notice_state,notice_error,last_error,attempts,next_attempt_at,action_id,created_at,completed_at FROM group_departures WHERE request_id=?1",[id],|r|Ok(json!({
        "request_id":r.get::<_,String>(0)?,"chat_id":r.get::<_,i64>(1)?,"title":r.get::<_,Option<String>>(2)?,"reason":r.get::<_,String>(3)?,"block_rejoin":r.get::<_,bool>(4)?,"state":r.get::<_,String>(5)?,"notice_state":r.get::<_,String>(6)?,"notice_error":r.get::<_,Option<String>>(7)?,"last_error":r.get::<_,Option<String>>(8)?,"attempts":r.get::<_,i64>(9)?,"next_attempt_at":r.get::<_,i64>(10)?,"action_id":r.get::<_,i64>(11)?,"created_at":r.get::<_,String>(12)?,"completed_at":r.get::<_,Option<String>>(13)?
    })))?)
}
fn snapshot(conn: &Connection, config: &Config, chat: i64) -> Result<Option<Value>> {
    let Some(title) = conn
        .query_row(
            "SELECT title FROM group_module_settings WHERE chat_id=?1",
            [chat],
            |r| r.get::<_, Option<String>>(0),
        )
        .optional()?
    else {
        return Ok(None);
    };
    let last = latest(conn, chat)?;
    let denied: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM banned_groups WHERE chat_id=?1)",
        [chat],
        |r| r.get(0),
    )?;
    let revision = json!([title, last, denied]).to_string();
    let latest = last.map(|id| job(conn, &id)).transpose()?;
    Ok(Some(
        json!({"chat_id":chat,"title":title,"revision":revision,"protected":protected(conn,config,chat)?,"service_denied":denied,"latest":latest}),
    ))
}

impl Runtime {
    pub(super) fn migrate_v37_to_v38(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch("CREATE TABLE IF NOT EXISTS group_departures (
            request_id TEXT PRIMARY KEY,chat_id INTEGER NOT NULL,actor_id INTEGER NOT NULL,payload TEXT NOT NULL,
            title TEXT,reason TEXT NOT NULL,block_rejoin INTEGER NOT NULL,
            state TEXT NOT NULL DEFAULT 'queued' CHECK(state IN ('queued','leaving','unconfirmed','done','failed','cancelled')),
            notice_state TEXT NOT NULL DEFAULT 'pending',notice_error TEXT,last_error TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,next_attempt_at INTEGER NOT NULL DEFAULT 0,
            action_id INTEGER NOT NULL,created_at TEXT NOT NULL,completed_at TEXT);
            CREATE INDEX IF NOT EXISTS idx_group_departure_due ON group_departures(state,next_attempt_at);
            CREATE INDEX IF NOT EXISTS idx_group_departure_chat ON group_departures(chat_id);
            PRAGMA user_version=38;")?;
        tx.commit()?;
        Ok(())
    }
    pub(super) async fn host_departure(&self, actor: i64, chat: i64) -> Result<Outcome> {
        if !is_host(actor) {
            return Ok(Outcome::Forbidden);
        }
        let config = self.config.clone();
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            let v = snapshot(&tx, &config, chat)?;
            tx.commit()?;
            Ok(v.map(Outcome::Ready).unwrap_or(Outcome::Invalid))
        })
        .await
    }
    pub(super) async fn queue_departure(&self, actor: i64, patch: Patch) -> Result<Outcome> {
        if !is_host(actor) {
            return Ok(Outcome::Forbidden);
        }
        if Uuid::parse_str(&patch.request_id).is_err()
            || patch.reason.trim().is_empty()
            || patch.reason.chars().count() > 500
            || patch.expected_revision.len() > 2048
        {
            return Ok(Outcome::Invalid);
        }
        let config = self.config.clone();
        let guard = self.user_action_guard(patch.chat_id).await;
        let mut cache = self.banned_groups.clone().write_owned().await;
        self.with_conn(move|conn|{
            let _guard=guard;let tx=conn.transaction()?;let payload=serde_json::to_string(&patch)?;
            if let Some((owner,old))=tx.query_row("SELECT actor_id,payload FROM group_departures WHERE request_id=?1",[&patch.request_id],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,String>(1)?))).optional()? {
                return Ok(if owner==actor && old==payload {Outcome::Ready(job(&tx,&patch.request_id)?)}else{Outcome::Conflict});
            }
            let Some(current)=snapshot(&tx,&config,patch.chat_id)? else {return Ok(Outcome::Invalid)};
            if current["protected"]==true {return Ok(Outcome::Forbidden)}
            if current["revision"]!=patch.expected_revision || matches!(current["latest"]["state"].as_str(),Some("queued"|"leaving"|"unconfirmed")) {return Ok(Outcome::Conflict)}
            let created=Utc::now().to_rfc3339();
            if patch.block_rejoin {
                tx.execute("INSERT INTO banned_groups(chat_id,reason,added_by,created_at) VALUES (?1,?2,?3,?4) ON CONFLICT(chat_id) DO NOTHING",params![patch.chat_id,patch.reason.trim(),actor,created])?;
            }
            let summary=json!({"reason":patch.reason.trim(),"block_rejoin":patch.block_rejoin,"request_id":patch.request_id});
            tx.execute("INSERT INTO maintainer_actions(actor_id,actor_name,chat_id,command,summary,undo_data,created_at) VALUES (?1,'項目主持人',?2,'退群',?3,?4,?5)",params![actor,patch.chat_id,summary.to_string(),serde_json::to_string(&UndoData::NotRevertible)?,created])?;
            let action=tx.last_insert_rowid();
            tx.execute("INSERT INTO group_departures(request_id,chat_id,actor_id,payload,title,reason,block_rejoin,action_id,created_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",params![patch.request_id,patch.chat_id,actor,payload,current["title"].as_str(),patch.reason.trim(),patch.block_rejoin,action,created])?;
            let result=job(&tx,&patch.request_id)?;tx.commit()?;
            if patch.block_rejoin {cache.insert(patch.chat_id);}
            Ok(Outcome::Ready(result))
        }).await
    }
}

type Guard = Arc<tokio::sync::OwnedMutexGuard<()>>;
async fn update(
    runtime: &Runtime,
    id: &str,
    state: &str,
    error: Option<String>,
    guard: Guard,
) -> Result<()> {
    let id = id.to_string();
    let state = state.to_string();
    runtime.with_conn(move|conn|{let _guard=guard;let tx=conn.transaction()?;
        tx.execute("UPDATE group_departures SET state=?2,last_error=?3,completed_at=CASE WHEN ?2 IN ('done','failed','cancelled') THEN ?4 ELSE NULL END WHERE request_id=?1",params![id,state,error,Utc::now().to_rfc3339()])?;
        tx.execute("UPDATE maintainer_actions SET summary=json_set(summary,'$.state',?2,'$.last_error',?3) WHERE action_id=(SELECT action_id FROM group_departures WHERE request_id=?1)",params![id,state,error])?;
        tx.commit()?;
        Ok(())
    }).await
}
async fn api<T>(
    request: impl std::future::Future<Output = ResponseResult<T>>,
) -> ResponseResult<T> {
    tokio::time::timeout(Duration::from_secs(20), request)
        .await
        .unwrap_or_else(|_| {
            Err(teloxide::RequestError::Io(
                std::io::Error::new(std::io::ErrorKind::TimedOut, "Telegram request timed out")
                    .into(),
            ))
        })
}
async fn failure(
    runtime: &Runtime,
    id: &str,
    state: &str,
    e: teloxide::RequestError,
    guard: Guard,
) -> Result<()> {
    if let teloxide::RequestError::RetryAfter(seconds) = &e {
        runtime.delay_telegram_queue(seconds.seconds()).await?;
    }
    update(
        runtime,
        id,
        state,
        Some(notices::diagnostic(&runtime.config, &e.to_string())),
        guard,
    )
    .await
}
pub(super) async fn attempt(bot: &Bot, runtime: &Runtime, id: &str) -> Result<()> {
    let id = id.to_string();
    let query = id.clone();
    let chat = runtime
        .with_conn(move |conn| {
            Ok(conn
                .query_row(
                    "SELECT chat_id FROM group_departures WHERE request_id=?1",
                    [query],
                    |r| r.get::<_, i64>(0),
                )
                .optional()?)
        })
        .await?;
    let Some(chat) = chat else { return Ok(()) };
    let guard = Arc::new(runtime.user_action_guard(chat).await);
    let held = guard.clone();
    let query = id.clone();
    let config = runtime.config.clone();
    let claimed=runtime.with_conn(move|conn|{
        let _guard=held;let tx=conn.transaction()?;let now=Utc::now().timestamp();
        if protected(&tx,&config,chat)? {
            tx.execute("UPDATE group_departures SET state='cancelled',last_error='This is a protected project group.' WHERE request_id=?1 AND state IN ('queued','leaving','unconfirmed')",[query])?;tx.commit()?;return Ok(None);
        }
        if tx.execute("UPDATE group_departures SET attempts=attempts+1,next_attempt_at=?2+MIN(3600,60*(1<<MIN(attempts,6))) WHERE request_id=?1 AND state IN ('queued','leaving','unconfirmed') AND next_attempt_at<=?2 AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?2",params![query,now])?==0 {return Ok(None)}
        let v=job(&tx,&query)?;tx.commit()?;Ok(Some(v))
    }).await?;
    let Some(saved) = claimed else { return Ok(()) };
    let state = saved["state"].as_str().unwrap();
    let Some(me) = tokio::time::timeout(Duration::from_secs(20), runtime.me_id(bot))
        .await
        .ok()
        .flatten()
    else {
        return update(
            runtime,
            &id,
            state,
            Some("Unable to identify the bot.".into()),
            guard,
        )
        .await;
    };
    let member = match api(bot.get_chat_member(ChatId(chat), me).into_future()).await {
        Ok(m) => m,
        Err(e) => return failure(runtime, &id, state, e, guard).await,
    };
    if matches!(
        member.kind,
        teloxide::types::ChatMemberKind::Left | teloxide::types::ChatMemberKind::Banned(_)
    ) {
        let query=id.clone();let held=guard.clone();
        runtime.with_conn(move|conn|{let _guard=held;conn.execute("UPDATE group_departures SET notice_state='skipped' WHERE request_id=?1 AND notice_state='pending'",[query])?;Ok(())}).await?;
        return update(runtime, &id, "done", None, guard).await;
    }
    if state != "queued" {
        return update(
            runtime,
            &id,
            "failed",
            Some(
                saved["last_error"]
                    .as_str()
                    .unwrap_or(
                        "The bot is still in this group. Confirm a new request to try again.",
                    )
                    .into(),
            ),
            guard,
        )
        .await;
    }
    // Save before sending: a lost response must not produce repeated notices.
    if saved["notice_state"] == "pending" {
        let query = id.clone();
        let held = guard.clone();
        runtime
            .with_conn(move |conn| {
                let _guard = held;
                conn.execute(
                    "UPDATE group_departures SET notice_state='unconfirmed' WHERE request_id=?1",
                    [query],
                )?;
                Ok(())
            })
            .await?;
        let text = format!(
            "SPB 將退出本群。\n原因：{}",
            saved["reason"].as_str().unwrap()
        );
        let sent = api(bot.send_message(ChatId(chat), text).into_future()).await;
        let (notice, error, cooldown) = match sent {
            Ok(_) => ("sent", None, None),
            Err(e) => {
                let delay = if let teloxide::RequestError::RetryAfter(s) = &e {
                    Some(s.seconds())
                } else {
                    None
                };
                (
                    if matches!(&e,teloxide::RequestError::Api(_)|teloxide::RequestError::RetryAfter(_)) {"failed"}else{"unconfirmed"},
                    Some(notices::diagnostic(&runtime.config, &e.to_string())),
                    delay,
                )
            }
        };
        let query = id.clone();
        let held = guard.clone();
        runtime.with_conn(move|conn|{let _guard=held;conn.execute("UPDATE group_departures SET notice_state=?2,notice_error=?3 WHERE request_id=?1",params![query,notice,error])?;Ok(())}).await?;
        if let Some(seconds) = cooldown {
            runtime.delay_telegram_queue(seconds).await?;
            return Ok(());
        }
    }
    // Never repeat a leave after a crash or lost acknowledgement. Reconcile
    // membership first; a fresh host confirmation is required to send again.
    update(runtime, &id, "leaving", None, guard.clone()).await?;
    match api(bot.leave_chat(ChatId(chat)).into_future()).await {
        Ok(_) => update(runtime, &id, "done", None, guard).await,
        Err(e) => failure(runtime, &id, "unconfirmed", e, guard).await,
    }
}
pub(super) async fn retry(bot: &Bot, runtime: &Runtime) -> Result<()> {
    let ids=runtime.with_conn(|conn|{let mut s=conn.prepare("SELECT request_id FROM group_departures WHERE state IN ('queued','leaving','unconfirmed') AND next_attempt_at<=?1 ORDER BY next_attempt_at LIMIT 10")?;let rows=s.query_map([Utc::now().timestamp()],|r|r.get::<_,String>(0))?;Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)}).await?;
    for id in ids {
        attempt(bot, runtime, &id).await?;
    }
    Ok(())
}

pub(super) fn spawn_worker(bot: Bot, runtime: Arc<Runtime>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            if let Err(e) = retry(&bot, &runtime).await {
                log::warn!(
                    "group departure queue: {}",
                    notices::diagnostic(&runtime.config, &e.to_string())
                );
            }
            sleep(Duration::from_secs(5)).await;
        }
    })
}
