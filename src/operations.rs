use super::*;
use rusqlite::OptionalExtension;
use serde_json::{json, Value};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Controls {
    pub automatic_new_paused: bool,
    pub automatic_pending_paused: bool,
    pub network_paused: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Read {}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Patch {
    pub request_id: String,
    pub expected_revision: i64,
    pub controls: Controls,
}
pub(super) enum Outcome {
    Ready(Value),
    Forbidden,
    Invalid,
    Conflict,
}

pub(super) fn controls(conn: &Connection) -> Result<Controls> {
    Ok(conn.query_row("SELECT automatic_new_paused,automatic_pending_paused,network_paused FROM operations_controls WHERE id=1",[],|r|Ok(Controls{automatic_new_paused:r.get(0)?,automatic_pending_paused:r.get(1)?,network_paused:r.get(2)?}))?)
}
pub(super) fn automatic(action: &ActionKind) -> bool {
    matches!(
        action,
        ActionKind::AutoBan
            | ActionKind::GuestBotBan
            | ActionKind::GuestInvokerBan
            | ActionKind::FloodMute
            | ActionKind::CmdCleanMute
    )
}

fn snapshot(conn: &Connection) -> Result<Value> {
    let (revision, changed_at, changed_by): (i64, Option<String>, Option<i64>) = conn.query_row(
        "SELECT revision,changed_at,changed_by FROM operations_controls WHERE id=1",
        [],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;
    let pending_bans:i64=conn.query_row("SELECT COUNT(*) FROM origin_ban_jobs j JOIN cases c ON c.id=j.case_id WHERE j.state='pending' AND j.ban_done=0 AND c.action IN ('auto_ban','guest_bot_ban','guest_invoker_ban')",[],|r|r.get(0))?;
    let pending_restrictions:i64=conn.query_row("SELECT COUNT(*) FROM restriction_jobs j JOIN cases c ON c.id=j.case_id WHERE j.state='pending' AND json_extract(j.payload,'$.step')='apply' AND c.action IN ('flood_mute','cmd_clean_mute')",[],|r|r.get(0))?;
    let pending_network: i64 = conn.query_row(
        "SELECT COUNT(*) FROM network_deliveries WHERE state='pending'",
        [],
        |r| r.get(0),
    )?;
    let captchas:i64=conn.query_row("SELECT COUNT(*) FROM captcha_jobs WHERE json_extract(payload,'$.state') IN ('prepare','waiting','kick')",[],|r|r.get(0))?;
    let cooldown: i64 = conn.query_row(
        "SELECT not_before FROM telegram_retry_state WHERE id=1",
        [],
        |r| r.get(0),
    )?;
    Ok(
        json!({"controls":controls(conn)?,"revision":revision,"changed_at":changed_at,"changed_by":changed_by,
        "pending_bans":pending_bans,"pending_restrictions":pending_restrictions,"pending_network":pending_network,"captchas":captchas,
        "telegram_not_before":cooldown,"updated_at":Utc::now().timestamp(),"version":env!("GIT_HASH"),
        "schema":conn.query_row("PRAGMA user_version",[],|r|r.get::<_,i64>(0))?}),
    )
}

impl Runtime {
    pub(super) fn migrate_v36_to_v37(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch("CREATE TABLE IF NOT EXISTS operations_controls (
            id INTEGER PRIMARY KEY CHECK(id=1),revision INTEGER NOT NULL DEFAULT 0,
            automatic_new_paused INTEGER NOT NULL DEFAULT 0 CHECK(automatic_new_paused IN (0,1)),
            automatic_pending_paused INTEGER NOT NULL DEFAULT 0 CHECK(automatic_pending_paused IN (0,1)),
            network_paused INTEGER NOT NULL DEFAULT 0 CHECK(network_paused IN (0,1)),
            captcha_epoch INTEGER NOT NULL DEFAULT 0,changed_at TEXT,changed_by INTEGER);
            INSERT OR IGNORE INTO operations_controls(id) VALUES(1);
            CREATE TABLE IF NOT EXISTS operations_requests(request_id TEXT PRIMARY KEY,actor_id INTEGER NOT NULL,payload TEXT NOT NULL,result TEXT NOT NULL,action_id INTEGER NOT NULL);
            PRAGMA user_version=37;")?;
        tx.commit()?;
        Ok(())
    }

    pub(super) async fn operations_controls(&self) -> Result<Controls> {
        self.with_conn(move |conn| controls(conn)).await
    }

    pub(super) async fn host_operations(&self, actor: i64) -> Result<Outcome> {
        if !is_host(actor) {
            return Ok(Outcome::Forbidden);
        }
        self.with_conn(|conn| {
            let tx = conn.transaction()?;
            let v = snapshot(&tx)?;
            tx.commit()?;
            Ok(Outcome::Ready(v))
        })
        .await
    }
    pub(super) async fn save_operations(&self, actor: i64, patch: Patch) -> Result<Outcome> {
        if !is_host(actor) {
            return Ok(Outcome::Forbidden);
        }
        if Uuid::parse_str(&patch.request_id).is_err() || patch.expected_revision < 0 {
            return Ok(Outcome::Invalid);
        }
        self.with_conn(move|conn| {
            let tx=conn.transaction()?;let payload=serde_json::to_string(&patch)?;
            if let Some((owner,old,result))=tx.query_row("SELECT actor_id,payload,result FROM operations_requests WHERE request_id=?1",[&patch.request_id],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?))).optional()? {
                return Ok(if owner==actor && old==payload {Outcome::Ready(serde_json::from_str(&result)?)}else{Outcome::Conflict});
            }
            let revision:i64=tx.query_row("SELECT revision FROM operations_controls WHERE id=1",[],|r|r.get(0))?;
            if revision!=patch.expected_revision {return Ok(Outcome::Conflict);}
            let before=controls(&tx)?;let after=&patch.controls;
            if &before==after {return Ok(Outcome::Invalid);}
            let now=Utc::now().timestamp();
            if before.network_paused && !after.network_paused {
                tx.execute("WITH due AS MATERIALIZED (SELECT case_id,chat_id,ROW_NUMBER() OVER(ORDER BY next_attempt_at,case_id,chat_id) AS position FROM network_deliveries WHERE state='pending')
                    UPDATE network_deliveries SET next_attempt_at=MAX(next_attempt_at,?1+3*(SELECT position FROM due WHERE due.case_id=network_deliveries.case_id AND due.chat_id=network_deliveries.chat_id)) WHERE state='pending'",[now])?;
            }
            if before.automatic_pending_paused && !after.automatic_pending_paused {
                tx.execute("WITH due AS MATERIALIZED (SELECT j.case_id,ROW_NUMBER() OVER(ORDER BY j.next_attempt_at,j.case_id) AS position FROM origin_ban_jobs j JOIN cases c ON c.id=j.case_id WHERE j.state='pending' AND j.ban_done=0 AND c.action IN ('auto_ban','guest_bot_ban','guest_invoker_ban'))
                    UPDATE origin_ban_jobs SET next_attempt_at=MAX(next_attempt_at,?1+3*(SELECT position FROM due WHERE due.case_id=origin_ban_jobs.case_id)) WHERE case_id IN (SELECT case_id FROM due)",[now])?;
                tx.execute("WITH due AS MATERIALIZED (SELECT j.case_id,ROW_NUMBER() OVER(ORDER BY j.next_attempt_at,j.case_id) AS position FROM restriction_jobs j JOIN cases c ON c.id=j.case_id WHERE j.state='pending' AND json_extract(j.payload,'$.step')='apply' AND c.action IN ('flood_mute','cmd_clean_mute'))
                    UPDATE restriction_jobs SET next_attempt_at=MAX(next_attempt_at,?1+3*(SELECT position FROM due WHERE due.case_id=restriction_jobs.case_id)) WHERE case_id IN (SELECT case_id FROM due)",[now])?;
            }
            tx.execute("UPDATE operations_controls SET revision=revision+1,automatic_new_paused=?1,automatic_pending_paused=?2,network_paused=?3,
                captcha_epoch=captcha_epoch+?4,changed_at=?5,changed_by=?6 WHERE id=1",
                params![after.automatic_new_paused,after.automatic_pending_paused,after.network_paused,!before.automatic_pending_paused && after.automatic_pending_paused,Utc::now().to_rfc3339(),actor])?;
            if !before.automatic_pending_paused && after.automatic_pending_paused {
                tx.execute("UPDATE captcha_jobs SET next_attempt_at=0 WHERE json_extract(payload,'$.state') IN ('prepare','waiting','kick')",[])?;
            }
            let summary=json!({"before":before,"after":after});
            tx.execute("INSERT INTO maintainer_actions(actor_id,actor_name,chat_id,command,summary,undo_data,created_at) VALUES (?1,'項目主持人',NULL,'緊急控制',?2,?3,?4)",params![actor,summary.to_string(),serde_json::to_string(&UndoData::NotRevertible)?,Utc::now().to_rfc3339()])?;
            let action_id=tx.last_insert_rowid();let mut result=snapshot(&tx)?;result["action_id"]=json!(action_id);
            tx.execute("INSERT INTO operations_requests(request_id,actor_id,payload,result,action_id) VALUES (?1,?2,?3,?4,?5)",params![patch.request_id,actor,payload,result.to_string(),action_id])?;
            tx.commit()?;Ok(Outcome::Ready(result))
        }).await
    }
}
