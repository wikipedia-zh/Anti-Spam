use super::*;
use rusqlite::OptionalExtension;

#[derive(Clone, Copy)]
pub(super) struct Observation {
    revision: i64,
    started_at: i64,
}

pub(super) fn observation(conn: &Connection, chat: i64) -> Result<Observation> {
    Ok(Observation {
        revision: snapshot(conn, chat)?.1,
        started_at: Utc::now().timestamp(),
    })
}

pub(super) fn snapshot(conn: &Connection, chat: i64) -> Result<(String, i64)> {
    Ok(conn
        .query_row(
            "SELECT state,revision FROM group_access WHERE chat_id=?1",
            [chat],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?
        .unwrap_or(("unknown".into(), 0)))
}

pub(super) fn blocked(conn: &Connection, chat: i64) -> Result<bool> {
    Ok(matches!(
        snapshot(conn, chat)?.0.as_str(),
        "left" | "unavailable"
    ))
}

fn error_state(error: &teloxide::RequestError) -> Option<&'static str> {
    use teloxide::{ApiError, RequestError};
    match error {
        RequestError::Api(ApiError::ChatNotFound) => Some("unavailable"),
        RequestError::Api(
            ApiError::BotKicked
            | ApiError::BotKickedFromSupergroup
            | ApiError::BotKickedFromChannel,
        ) => Some("left"),
        RequestError::Api(ApiError::Unknown(message))
            if matches!(
                message.as_str(),
                "Forbidden: bot is not a member of the supergroup chat"
                    | "Forbidden: bot is not a member of the channel chat"
                    | "Forbidden: bot was kicked from the group chat"
            ) =>
        {
            Some("left")
        }
        _ => None,
    }
}

// Keep unfinished work and uncertain outcomes. Rejoining only makes it due;
// each worker still checks the current case, whitelist and group settings.
fn resume(tx: &rusqlite::Transaction<'_>, chat: i64, now: i64) -> Result<()> {
    for (table, order) in [
        ("network_deliveries", "case_id"),
        ("network_catchups", "message_id,user_id"),
    ] {
        tx.execute(&format!("WITH due AS MATERIALIZED (SELECT rowid,ROW_NUMBER() OVER (ORDER BY {order}) AS n
            FROM {table} WHERE chat_id=?1 AND state='pending')
            UPDATE {table} SET next_attempt_at=MAX(next_attempt_at,?2,
                (SELECT not_before FROM telegram_retry_state WHERE id=1))+(SELECT n FROM due WHERE due.rowid={table}.rowid)*3
            WHERE rowid IN (SELECT rowid FROM due)"),params![chat,now])?;
    }
    Ok(())
}

impl Runtime {
    pub(super) fn migrate_v38_to_v39(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS group_access (
                chat_id INTEGER PRIMARY KEY,state TEXT NOT NULL DEFAULT 'unknown'
                    CHECK(state IN ('unknown','present','left','unavailable')),
                revision INTEGER NOT NULL DEFAULT 0,event_date INTEGER NOT NULL DEFAULT 0,
                event_id INTEGER NOT NULL DEFAULT -1,checked_at INTEGER NOT NULL DEFAULT 0,
                next_check_at INTEGER NOT NULL DEFAULT 0);
            PRAGMA user_version=39;",
        )?;
        tx.commit()?;
        Ok(())
    }

    pub(super) async fn group_access_revision(&self, chat: i64) -> Result<Observation> {
        self.with_conn(move |conn| observation(conn, chat)).await
    }

    pub(super) async fn record_group_access_error(
        &self,
        chat: i64,
        observation: Observation,
        error: &teloxide::RequestError,
    ) -> Result<()> {
        let Some(state) = error_state(error) else {
            return Ok(());
        };
        self.record_group_access_check(chat, observation, state)
            .await
    }

    pub(super) async fn record_group_access_check(
        &self,
        chat: i64,
        observation: Observation,
        state: &'static str,
    ) -> Result<()> {
        if chat >= 0 {
            return Ok(());
        }
        self.with_conn(move |conn| {
            let tx=conn.transaction()?;let now=Utc::now().timestamp();
            let (old,current)=snapshot(&tx,chat)?;
            // A delayed API failure must not overwrite a newer membership update.
            if current!=observation.revision {return Ok(());}
            tx.execute("INSERT OR IGNORE INTO group_access(chat_id) VALUES (?1)",[chat])?;
            tx.execute("UPDATE group_access SET state=?2,revision=revision+1,checked_at=?3,next_check_at=?4+3600 WHERE chat_id=?1",params![chat,state,observation.started_at,now])?;
            if state=="present" && matches!(old.as_str(),"left"|"unavailable") {resume(&tx,chat,now)?;}
            tx.commit()?;Ok(())
        }).await
    }

    pub(super) async fn record_bot_membership(
        &self,
        update_id: u32,
        member: ChatMemberUpdated,
    ) -> Result<()> {
        if !member.chat.is_group() && !member.chat.is_supergroup() {
            return Ok(());
        }
        let chat = member.chat.id.0;
        let date = member.date.timestamp();
        let state = if member.new_chat_member.kind.is_present() {
            "present"
        } else {
            "left"
        };
        let title = member.chat.title().map(str::to_owned);
        self.with_conn(move |conn| {
            let tx=conn.transaction()?;let now=Utc::now().timestamp();
            tx.execute("INSERT OR IGNORE INTO group_access(chat_id) VALUES (?1)",[chat])?;
            let (old,checked,event_date,event_id):(String,i64,i64,i64)=tx.query_row("SELECT state,checked_at,event_date,event_id FROM group_access WHERE chat_id=?1",[chat],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)))?;
            if (date,i64::from(update_id))<=(event_date,event_id) {return Ok(());}
            // Events delivered after a newer API observation still advance the
            // replay cursor, but must not undo that observation.
            let applied=if date<checked {old.as_str()}else{state};
            tx.execute("UPDATE group_access SET state=?2,revision=revision+1,event_date=?3,event_id=?4,next_check_at=?5+3600 WHERE chat_id=?1",params![chat,applied,date,update_id,now])?;
            tx.execute("INSERT INTO group_module_settings(chat_id,title,no_contact) VALUES (?1,?2,1)
                ON CONFLICT(chat_id) DO UPDATE SET title=COALESCE(excluded.title,group_module_settings.title)",params![chat,title])?;
            if applied=="present" && matches!(old.as_str(),"left"|"unavailable") {resume(&tx,chat,now)?;}
            tx.commit()?;Ok(())
        }).await
    }
}

// One membership check per hour for inaccessible chats also recovers missed
// rejoin events. Old delivery errors get a fresh check, not a guessed status.
pub(super) async fn reconcile(bot: &Bot, runtime: &Runtime) -> Result<()> {
    let candidates = runtime
        .with_conn(|conn| {
            let mut s = conn.prepare(
                "SELECT chat_id FROM group_access WHERE state!='present' AND next_check_at<=?1
            UNION SELECT DISTINCT d.chat_id FROM network_deliveries d
                WHERE d.state='pending' AND d.last_error IS NOT NULL
                AND NOT EXISTS(SELECT 1 FROM group_access a WHERE a.chat_id=d.chat_id)
            LIMIT 5",
            )?;
            let rows = s.query_map([Utc::now().timestamp()], |r| r.get::<_, i64>(0))?;
            Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
        })
        .await?;
    if candidates.is_empty() {
        return Ok(());
    }
    let Some(me) = tokio::time::timeout(Duration::from_secs(30), runtime.me_id(bot))
        .await
        .ok()
        .flatten()
    else {
        return Ok(());
    };
    for chat in candidates {
        let claim=runtime.with_conn(move |conn| {
            let tx=conn.transaction()?;let now=Utc::now().timestamp();
            let cooldown:i64=tx.query_row("SELECT not_before FROM telegram_retry_state WHERE id=1",[],|r|r.get(0))?;
            if cooldown>now {return Ok(None);}
            tx.execute("INSERT OR IGNORE INTO group_access(chat_id) VALUES (?1)",[chat])?;
            if tx.execute("UPDATE group_access SET next_check_at=?2+3600 WHERE chat_id=?1 AND state!='present' AND next_check_at<=?2",params![chat,now])?==0 {return Ok(None);}
            let observation=observation(&tx,chat)?;tx.commit()?;Ok(Some(observation))
        }).await?;
        let Some(revision) = claim else {
            continue;
        };
        match tokio::time::timeout(Duration::from_secs(30), async {
            bot.get_chat_member(ChatId(chat), me).await
        })
        .await
        {
            Ok(Ok(member)) => {
                runtime
                    .record_group_access_check(
                        chat,
                        revision,
                        if member.kind.is_present() {
                            "present"
                        } else {
                            "left"
                        },
                    )
                    .await?
            }
            Ok(Err(teloxide::RequestError::RetryAfter(delay))) => {
                runtime.delay_telegram_queue(delay.seconds()).await?;
                break;
            }
            Ok(Err(error)) => {
                runtime
                    .record_group_access_error(chat, revision, &error)
                    .await?
            }
            Err(_) => {}
        }
    }
    Ok(())
}
