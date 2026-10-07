use super::*;
use rusqlite::OptionalExtension;

#[derive(Serialize, Deserialize)]
struct Entry {
    id: i64,
    handle: String,
    pattern: String,
}

struct Guards {
    _job: tokio::sync::OwnedMutexGuard<()>,
    rules: tokio::sync::OwnedRwLockReadGuard<Vec<SpamRule>>,
}

impl Runtime {
    pub(super) fn migrate_v28_to_v29(conn: &mut Connection) -> Result<()> {
        conn.execute_batch("BEGIN;
            CREATE TABLE IF NOT EXISTS rule_captures(case_id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS rule_notice_jobs(
                id TEXT PRIMARY KEY,case_id TEXT NOT NULL,chat_id INTEGER NOT NULL,source_chat_id INTEGER NOT NULL,
                payload TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'pending',message_id INTEGER,
                attempts INTEGER NOT NULL DEFAULT 0,next_attempt_at INTEGER NOT NULL DEFAULT 0,last_error TEXT);
            CREATE INDEX IF NOT EXISTS idx_rule_notice_due ON rule_notice_jobs(state,next_attempt_at);
            PRAGMA user_version=29; COMMIT;")?;
        Ok(())
    }

    pub(super) async fn capture_rules(&self, case: &CaseRecord) -> Result<bool> {
        let Some(handles) = bot_mentions_only(&case.evidence_text) else {
            return Ok(false);
        };
        let case = case.clone();
        let log_chat = self.config.log_channel_id;
        self.with_rule_transaction(move |tx| {
            if tx.execute("INSERT OR IGNORE INTO rule_captures(case_id) VALUES (?1)",[&case.id])?==0 {return Ok(true);}
            let mut created=Vec::new();
            for handle in handles {
                let pattern=format!("(?i)@{handle}\\b");
                let exists=tx.query_row("SELECT EXISTS(SELECT 1 FROM spam_rules WHERE pattern=?1 COLLATE NOCASE)",[&pattern],|r|r.get::<_,bool>(0))?;
                if exists {continue;}
                FancyRegex::new(&pattern).context("invalid bot rule")?;
                tx.execute("INSERT INTO spam_rules(pattern,description) VALUES (?1,?2)",params![pattern,format!("純機器人提及 spam：@{handle}（自動建立）")])?;
                created.push(Entry{id:tx.last_insert_rowid(),handle,pattern});
            }
            for (batch,entries) in created.chunks(20).enumerate() {
                tx.execute("INSERT INTO rule_notice_jobs(id,case_id,chat_id,source_chat_id,payload) VALUES (?1,?2,?3,?4,?5)",
                    params![format!("{}:{batch}",case.id),case.id,log_chat,case.chat_id,serde_json::to_string(entries)?])?;
            }
            Ok(true)
        }).await
    }
}

pub(super) async fn attempt(bot: &Bot, runtime: &Runtime, id: &str) -> Result<()> {
    let guards = Arc::new(Guards {
        _job: runtime.review_guard(&format!("rule-notice:{id}")).await,
        rules: runtime.spam_rules.clone().read_owned().await,
    });
    let id = id.to_string();
    let claim_id = id.clone();
    let claim_guard = guards.clone();
    let row=runtime.with_conn(move |conn| {
        let _guard=claim_guard;let tx=conn.transaction()?;let now=Utc::now().timestamp();
        let row=tx.query_row("SELECT chat_id,source_chat_id,payload,attempts FROM rule_notice_jobs WHERE id=?1 AND state='pending' AND next_attempt_at<=?2
            AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?2",params![claim_id,now],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,String>(2)?,r.get::<_,u32>(3)?))).optional()?;
        if let Some((_,_,_,attempts))=&row {
            let n=attempts.saturating_add(1).min(31);
            tx.execute("UPDATE rule_notice_jobs SET attempts=?2,next_attempt_at=?3 WHERE id=?1",params![claim_id,n,now+(60_i64<<n.saturating_sub(1).min(6)).min(3600)])?;
        }
        tx.commit()?;Ok(row)
    }).await?;
    let Some((chat, source, payload, _)) = row else {
        return Ok(());
    };
    let entries: Vec<Entry> = serde_json::from_str(&payload)?;
    // Keep a deleted or edited rule out of a delayed creation notice.
    let current = entries
        .iter()
        .filter(|entry| {
            guards
                .rules
                .iter()
                .any(|rule| rule.id == entry.id && rule.regex.as_str() == entry.pattern)
        })
        .collect::<Vec<_>>();
    if current.is_empty() {
        return runtime
            .with_conn(move |conn| {
                let _guard = guards;
                conn.execute(
                    "UPDATE rule_notice_jobs SET state='cancelled',last_error=NULL WHERE id=?1",
                    [id],
                )?;
                Ok(())
            })
            .await;
    }
    let list = current
        .iter()
        .map(|entry| {
            format!(
                "@{}（規則 #{}）",
                escape_html(&notices::preview(&entry.handle, 64)),
                entry.id
            )
        })
        .collect::<Vec<_>>()
        .join("、");
    let text=format!("<b>已自動建立機器人提及規則</b>\n來源：群組 <code>{source}</code>\n{list}\n如為誤判請用 /del_rule 移除。");
    let result:Result<i32>=async {
        let sent=tokio::time::timeout(Duration::from_secs(30),async {bot.send_message(ChatId(chat),text).parse_mode(ParseMode::Html).await}).await.context("Telegram request timed out")??;
        let message_id=sent.id.0;let ack_id=id.clone();let ack_guard=guards.clone();
        runtime.with_conn(move |conn| {let _guard=ack_guard;conn.execute("UPDATE rule_notice_jobs SET state='done',message_id=?2,last_error=NULL WHERE id=?1",params![ack_id,message_id])?;Ok(())}).await?;
        Ok(message_id)
    }.await;
    if let Err(error) = result {
        if let Some(teloxide::RequestError::RetryAfter(delay)) =
            error.downcast_ref::<teloxide::RequestError>()
        {
            runtime.delay_telegram_queue(delay.seconds()).await?;
        }
        let diagnostic = notices::diagnostic(&runtime.config, &error.to_string())
            .chars()
            .take(1000)
            .collect::<String>();
        log::warn!("rule notice {id}: {diagnostic}");
        runtime
            .with_conn(move |conn| {
                let _guard = guards;
                conn.execute(
                    "UPDATE rule_notice_jobs SET last_error=?2 WHERE id=?1",
                    params![id, diagnostic],
                )?;
                Ok(())
            })
            .await?;
    }
    Ok(())
}

pub(super) async fn retry(bot: &Bot, runtime: &Runtime) -> Result<()> {
    let ids=runtime.with_conn(|conn| {
        let mut stmt=conn.prepare("SELECT id FROM rule_notice_jobs WHERE state='pending' AND next_attempt_at<=?1 AND (SELECT not_before FROM telegram_retry_state WHERE id=1)<=?1 ORDER BY next_attempt_at,id LIMIT 20")?;
        let rows=stmt.query_map([Utc::now().timestamp()],|r|r.get::<_,String>(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }).await?;
    for id in ids {
        if let Err(error) = attempt(bot, runtime, &id).await {
            log::warn!(
                "rule notice {id}: {}",
                notices::diagnostic(&runtime.config, &error.to_string())
            );
        }
    }
    Ok(())
}

pub(super) async fn capture(bot: &Bot, runtime: &Runtime, case: &CaseRecord) -> Result<bool> {
    let captured = runtime.capture_rules(case).await?;
    if captured {
        let case_id = case.id.clone();
        let id=runtime.with_conn(move |conn| Ok(conn.query_row("SELECT id FROM rule_notice_jobs WHERE case_id=?1 AND state='pending' ORDER BY id LIMIT 1",[case_id],|r|r.get::<_,String>(0)).optional()?)).await?;
        if let Some(id) = id {
            attempt(bot, runtime, &id).await?;
        }
    }
    Ok(captured)
}
