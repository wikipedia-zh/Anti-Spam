use super::*;
use rusqlite::OptionalExtension;
use serde_json::{json, Value};

static RULE_WORK: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(2);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Rule {
    pub pattern: String,
    pub description: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Target {
    pub rule_id: Option<i64>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Patch {
    pub request_id: String,
    pub rule_id: Option<i64>,
    pub expected_revision: i64,
    pub rule: Option<Rule>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Trial {
    pub pattern: String,
    pub text: String,
}
pub(super) enum Outcome {
    Saved(Value),
    Conflict,
    Invalid,
    Forbidden,
    Busy,
}
fn valid_pattern(pattern: &str) -> bool {
    !pattern.trim().is_empty() && pattern.chars().count() <= 512
}
fn compile(pattern: &str) -> Result<FancyRegex> {
    Ok(fancy_regex::RegexBuilder::new(pattern)
        .delegate_size_limit(2 * 1024 * 1024)
        .delegate_dfa_size_limit(2 * 1024 * 1024)
        .build()?)
}
fn rule(conn: &Connection, id: i64) -> Result<Option<Rule>> {
    Ok(conn
        .query_row(
            "SELECT pattern,description FROM spam_rules WHERE id=?1",
            [id],
            |r| {
                Ok(Rule {
                    pattern: r.get(0)?,
                    description: r.get(1)?,
                })
            },
        )
        .optional()?)
}
fn snapshot(conn: &Connection, id: Option<i64>) -> Result<Value> {
    let revision: i64 =
        conn.query_row("SELECT revision FROM rule_revision WHERE id=1", [], |r| {
            r.get(0)
        })?;
    let saved = match id {
        Some(id) => rule(conn, id)?,
        None => None,
    };
    Ok(json!({"rule_id":id,"revision":revision,"rule":saved}))
}

impl Runtime {
    pub(super) fn migrate_v33_to_v34(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch("CREATE TABLE IF NOT EXISTS rule_revision(id INTEGER PRIMARY KEY CHECK(id=1),revision INTEGER NOT NULL DEFAULT 0);
            INSERT OR IGNORE INTO rule_revision(id) VALUES (1);
            CREATE TABLE IF NOT EXISTS host_rule_requests(request_id TEXT PRIMARY KEY,actor_id INTEGER NOT NULL,payload TEXT NOT NULL,result TEXT NOT NULL,action_id INTEGER NOT NULL);")?;
        for op in ["INSERT", "UPDATE", "DELETE"] {
            tx.execute_batch(&format!("CREATE TRIGGER IF NOT EXISTS rules_revision_{op} AFTER {op} ON spam_rules BEGIN UPDATE rule_revision SET revision=revision+1 WHERE id=1; END;"))?;
        }
        tx.execute_batch("PRAGMA user_version=34;")?;
        tx.commit()?;
        Ok(())
    }

    pub(super) async fn host_rule(&self, actor: i64, target: Target) -> Result<Outcome> {
        if !is_host(actor) {
            return Ok(Outcome::Forbidden);
        }
        if target.rule_id.is_some_and(|id| id <= 0) {
            return Ok(Outcome::Invalid);
        }
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            let result = snapshot(&tx, target.rule_id)?;
            tx.commit()?;
            Ok(Outcome::Saved(result))
        })
        .await
    }

    pub(super) async fn test_host_rule(&self, actor: i64, trial: Trial) -> Result<Outcome> {
        if !is_host(actor) {
            return Ok(Outcome::Forbidden);
        }
        if !valid_pattern(&trial.pattern) || trial.text.chars().count() > 4000 {
            return Ok(Outcome::Invalid);
        }
        let Ok(permit) = RULE_WORK.try_acquire() else {
            return Ok(Outcome::Busy);
        };
        tokio::task::spawn_blocking(move|| {
            let _permit=permit;
            let result=match compile(&trial.pattern) {
                Err(_)=>json!({"status":"invalid"}),
                Ok(regex)=>match regex.find(&trial.text) {
                    Err(_)=>json!({"status":"limit"}),
                    Ok(None)=>json!({"status":"no_match"}),
                    Ok(Some(found))=>json!({"status":"match","matched":found.as_str(),"start":trial.text[..found.start()].chars().count(),"end":trial.text[..found.end()].chars().count()}),
                }
            };
            Ok(Outcome::Saved(result))
        }).await?
    }

    pub(super) async fn save_host_rule(&self, actor: i64, patch: Patch) -> Result<Outcome> {
        if !is_host(actor) {
            return Ok(Outcome::Forbidden);
        }
        if Uuid::parse_str(&patch.request_id).is_err()
            || patch.expected_revision < 0
            || patch.rule_id.is_some_and(|id| id <= 0)
            || (patch.rule_id.is_none() && patch.rule.is_none())
            || patch
                .rule
                .as_ref()
                .is_some_and(|r| !valid_pattern(&r.pattern) || r.description.chars().count() > 256)
        {
            return Ok(Outcome::Invalid);
        }
        if let Some(r) = &patch.rule {
            let Ok(permit) = RULE_WORK.try_acquire() else {
                return Ok(Outcome::Busy);
            };
            let pattern = r.pattern.clone();
            if !tokio::task::spawn_blocking(move || {
                let _permit = permit;
                compile(&pattern).is_ok()
            })
            .await?
            {
                return Ok(Outcome::Invalid);
            }
        }
        self.with_rule_transaction(move|tx| {
            let payload=serde_json::to_string(&patch)?;
            let saved=tx.query_row("SELECT actor_id,payload,result FROM host_rule_requests WHERE request_id=?1",[&patch.request_id],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?))).optional()?;
            if let Some((owner,previous,result))=saved {
                return Ok(if owner==actor && previous==payload {Outcome::Saved(serde_json::from_str(&result)?)}else{Outcome::Conflict});
            }
            let revision:i64=tx.query_row("SELECT revision FROM rule_revision WHERE id=1",[],|r|r.get(0))?;
            if revision!=patch.expected_revision {return Ok(Outcome::Conflict);}
            let before=match patch.rule_id {Some(id)=>rule(tx,id)?,None=>None};
            if patch.rule_id.is_some() && before.is_none() {return Ok(Outcome::Conflict);}
            if before==patch.rule {return Ok(Outcome::Invalid);}
            let (id,command,undo)=match (patch.rule_id,&before,&patch.rule) {
                (None,None,Some(after))=>{
                    tx.execute("INSERT INTO spam_rules(pattern,description) VALUES (?1,?2)",params![after.pattern,after.description])?;
                    let id=tx.last_insert_rowid();(id,"/add_rule",UndoData::RuleAdded{rule_id:id})
                },
                (Some(id),Some(before),Some(after))=>{
                    tx.execute("UPDATE spam_rules SET pattern=?2,description=?3 WHERE id=?1",params![id,after.pattern,after.description])?;
                    (id,"/edit_rule",UndoData::RuleUpdated{rule_id:id,before:before.clone(),after:after.clone()})
                },
                (Some(id),Some(before),None)=>{
                    tx.execute("DELETE FROM spam_rules WHERE id=?1",[id])?;
                    (id,"/del_rule",UndoData::RuleDeleted{pattern:before.pattern.clone(),description:before.description.clone()})
                },
                _=>return Ok(Outcome::Invalid),
            };
            let summary=json!({"rule_id":id,"before":before,"after":patch.rule});
            tx.execute("INSERT INTO maintainer_actions(actor_id,actor_name,chat_id,command,summary,undo_data,created_at) VALUES (?1,'項目主持人',NULL,?2,?3,?4,?5)",params![actor,command,summary.to_string(),serde_json::to_string(&undo)?,Utc::now().to_rfc3339()])?;
            let action_id=tx.last_insert_rowid();let result=snapshot(tx,Some(id))?;
            tx.execute("INSERT INTO host_rule_requests(request_id,actor_id,payload,result,action_id) VALUES (?1,?2,?3,?4,?5)",params![patch.request_id,actor,payload,result.to_string(),action_id])?;
            Ok(Outcome::Saved(result))
        }).await
    }

    pub(super) async fn restore_host_rule(
        &self,
        id: i64,
        before: Rule,
        after: Rule,
    ) -> Result<bool> {
        self.with_rule_transaction(move|tx|Ok(tx.execute("UPDATE spam_rules SET pattern=?2,description=?3 WHERE id=?1 AND pattern=?4 AND description=?5",params![id,before.pattern,before.description,after.pattern,after.description])?!=0)).await
    }
}
