use super::*;
use rusqlite::OptionalExtension;
use serde_json::{json, Value};
use std::collections::BTreeMap;

type Frequencies = BTreeMap<String, (u64, u64)>;

#[derive(Default)]
pub(super) struct Candidate {
    words: Frequencies,
    pub spam_docs: usize,
    pub ham_docs: usize,
}

#[derive(Serialize, Deserialize)]
struct Backup {
    words: Frequencies,
    meta: Vec<(String, Option<String>)>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Preview {}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Patch {
    pub request_id: String,
    pub expected_revision: String,
}

pub(super) enum Outcome {
    Saved(Value),
    Invalid,
    Forbidden,
    Busy,
    Conflict,
    TooLarge,
}

fn revision(conn: &Connection) -> Result<String> {
    let value: i64 = conn.query_row("SELECT revision FROM model_revision WHERE id=1", [], |r| {
        r.get(0)
    })?;
    Ok(format!("{}:{value}", env!("GIT_HASH")))
}

pub(super) fn samples(conn: &Connection) -> Result<Vec<(String, String)>> {
    let mut stmt =
        conn.prepare("SELECT label,text FROM training_samples WHERE trim(text)!='' ORDER BY id")?;
    let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

pub(super) fn candidate(samples: Vec<(String, String)>) -> Candidate {
    let mut result = Candidate::default();
    for (label, text) in samples {
        let spam = match label.as_str() {
            "spam" => {
                result.spam_docs += 1;
                true
            }
            "ham" => {
                result.ham_docs += 1;
                false
            }
            _ => continue,
        };
        for word in tokenize(&text) {
            let counts = result.words.entry(word).or_default();
            if spam {
                counts.0 += 1;
            } else {
                counts.1 += 1;
            }
        }
    }
    result
}

fn replace_words(tx: &rusqlite::Transaction<'_>, words: &Frequencies) -> Result<()> {
    tx.execute("DELETE FROM word_frequencies", [])?;
    let mut stmt =
        tx.prepare("INSERT INTO word_frequencies(word,spam_count,ham_count) VALUES (?1,?2,?3)")?;
    for (word, (spam, ham)) in words {
        stmt.execute(params![word, spam, ham])?;
    }
    Ok(())
}

pub(super) fn write(tx: &rusqlite::Transaction<'_>, model: &Candidate) -> Result<()> {
    replace_words(tx, &model.words)?;
    for (key, count) in [("spam_docs", model.spam_docs), ("ham_docs", model.ham_docs)] {
        tx.execute("INSERT INTO model_meta(key,value) VALUES (?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",params![key,count.to_string()])?;
    }
    Ok(())
}

fn stats(words: &Frequencies) -> Value {
    json!({"words":words.len(),"spam_total":words.values().map(|v|v.0).sum::<u64>(),"ham_total":words.values().map(|v|v.1).sum::<u64>()})
}

fn receipt(conn: &Connection, actor: i64, request: &Patch) -> Result<Option<Outcome>> {
    let saved: Option<(i64, String, String)> = conn
        .query_row(
            "SELECT actor_id,payload,result FROM model_rebuild_requests WHERE request_id=?1",
            [&request.request_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    saved
        .map(|(owner, payload, result)| {
            Ok(
                if owner == actor && payload == serde_json::to_string(request)? {
                    Outcome::Saved(serde_json::from_str(&result)?)
                } else {
                    Outcome::Conflict
                },
            )
        })
        .transpose()
}

struct Prepared {
    backup: Backup,
    candidate: Candidate,
    preview: Value,
}

impl Runtime {
    pub(super) fn migrate_v34_to_v35(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch("CREATE TABLE IF NOT EXISTS model_revision(id INTEGER PRIMARY KEY CHECK(id=1),revision INTEGER NOT NULL DEFAULT 0);
            INSERT OR IGNORE INTO model_revision(id) VALUES (1);
            CREATE TABLE IF NOT EXISTS model_rebuild_requests(request_id TEXT PRIMARY KEY,actor_id INTEGER NOT NULL,payload TEXT NOT NULL,result TEXT NOT NULL,backup_json TEXT NOT NULL,after_revision TEXT NOT NULL,action_id INTEGER NOT NULL,restored INTEGER NOT NULL DEFAULT 0);")?;
        for table in ["training_samples", "word_frequencies"] {
            for op in ["INSERT", "UPDATE", "DELETE"] {
                tx.execute_batch(&format!("CREATE TRIGGER IF NOT EXISTS model_revision_{table}_{op} AFTER {op} ON {table} BEGIN UPDATE model_revision SET revision=revision+1 WHERE id=1; END;"))?;
            }
        }
        for (op,condition) in [("INSERT","NEW.key IN ('spam_docs','ham_docs','spam_threshold')"),("DELETE","OLD.key IN ('spam_docs','ham_docs','spam_threshold')"),("UPDATE","NEW.key IN ('spam_docs','ham_docs','spam_threshold') OR OLD.key IN ('spam_docs','ham_docs','spam_threshold')")] {
            tx.execute_batch(&format!("CREATE TRIGGER IF NOT EXISTS model_revision_meta_{op} AFTER {op} ON model_meta WHEN {condition} BEGIN UPDATE model_revision SET revision=revision+1 WHERE id=1; END;"))?;
        }
        tx.execute_batch("PRAGMA user_version=35;")?;
        tx.commit()?;
        Ok(())
    }

    async fn prepare_model_rebuild(
        &self,
        permit: tokio::sync::SemaphorePermit<'static>,
    ) -> Result<(Option<Prepared>, tokio::sync::SemaphorePermit<'static>)> {
        let (snapshot,permit)=self.with_conn(move |conn| {
            let tx=conn.transaction()?;
            let (rows,bytes,largest):(i64,i64,i64)=tx.query_row("SELECT COUNT(*),COALESCE(SUM(length(CAST(text AS BLOB))),0),COALESCE(MAX(length(CAST(text AS BLOB))),0) FROM training_samples",[],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?;
            let (words,word_bytes):(i64,i64)=tx.query_row("SELECT COUNT(*),COALESCE(SUM(length(CAST(word AS BLOB))),0) FROM word_frequencies",[],|r|Ok((r.get(0)?,r.get(1)?)))?;
            if rows>20_000||bytes>16*1024*1024||largest>64*1024||words>200_000||word_bytes>16*1024*1024{return Ok((None,permit));}
            let mut frequencies=Frequencies::new();
            {let mut stmt=tx.prepare("SELECT word,spam_count,ham_count FROM word_frequencies")?;
             let stored=stmt.query_map([],|r|Ok((r.get::<_,String>(0)?,(r.get::<_,u64>(1)?,r.get::<_,u64>(2)?))))?;
             for row in stored {let (word,counts)=row?;frequencies.insert(word,counts);}}
            let mut meta=Vec::new();
            for key in ["spam_docs","ham_docs"] {meta.push((key.into(),tx.query_row("SELECT value FROM model_meta WHERE key=?1",[key],|r|r.get(0)).optional()?));}
            let source=samples(&tx)?;let rev=revision(&tx)?;let now=Utc::now().timestamp();tx.commit()?;
            Ok((Some((Backup{words:frequencies,meta},source,rev,rows,now)),permit))
        }).await?;
        let Some((backup, source, revision, rows, now)) = snapshot else {
            return Ok((None, permit));
        };
        tokio::task::spawn_blocking(move || {
            let candidate=candidate(source);
            if candidate.words.len()>200_000 {return Ok((None,permit));}
            let added=candidate.words.keys().filter(|w|!backup.words.contains_key(*w)).count();
            let removed=backup.words.keys().filter(|w|!candidate.words.contains_key(*w)).count();
            let changed=candidate.words.iter().filter(|(w,c)|backup.words.get(*w).is_some_and(|old|old!=*c)).count();
            let preview=json!({"revision":revision,"updated_at":now,"sample_rows":rows,"spam_docs":candidate.spam_docs,"ham_docs":candidate.ham_docs,
                "before":stats(&backup.words),"after":stats(&candidate.words),"changes":{"added":added,"removed":removed,"changed":changed}});
            Ok((Some(Prepared{backup,candidate,preview}),permit))
        }).await?
    }

    pub(super) async fn preview_model_rebuild(&self, actor: i64) -> Result<Outcome> {
        if !is_host(actor) {
            return Ok(Outcome::Forbidden);
        }
        let Ok(permit) = host_model::MODEL_WORK.try_acquire() else {
            return Ok(Outcome::Busy);
        };
        let (result, permit) = self.prepare_model_rebuild(permit).await?;
        drop(permit);
        Ok(match result {
            Some(p) => Outcome::Saved(p.preview),
            None => Outcome::TooLarge,
        })
    }

    pub(super) async fn save_model_rebuild(&self, actor: i64, request: Patch) -> Result<Outcome> {
        if !is_host(actor) {
            return Ok(Outcome::Forbidden);
        }
        if Uuid::parse_str(&request.request_id).is_err()
            || request.expected_revision.len() > 100
            || request.expected_revision.is_empty()
        {
            return Ok(Outcome::Invalid);
        }
        let first = request.clone();
        if let Some(saved) = self
            .with_conn(move |conn| receipt(conn, actor, &first))
            .await?
        {
            return Ok(saved);
        }
        let Ok(permit) = host_model::MODEL_WORK.try_acquire() else {
            return Ok(Outcome::Busy);
        };
        let (prepared, permit) = self.prepare_model_rebuild(permit).await?;
        let Some(prepared) = prepared else {
            return Ok(Outcome::TooLarge);
        };
        if prepared.preview["revision"] != request.expected_revision {
            return Ok(Outcome::Conflict);
        }
        self.with_model_transaction(move|tx| {
            let _permit=permit;
            if let Some(saved)=receipt(tx,actor,&request)?{return Ok(saved);}
            if revision(tx)?!=request.expected_revision{return Ok(Outcome::Conflict);}
            write(tx,&prepared.candidate)?;
            let after_revision=revision(tx)?;
            let undo=UndoData::ModelRebuilt{request_id:request.request_id.clone()};
            let before=&prepared.preview["before"];let after=&prepared.preview["after"];let changes=&prepared.preview["changes"];
            let summary=format!("詞彙 {} → {}；垃圾詞頻 {} → {}；正常詞頻 {} → {}；新增 {}、移除 {}、調整 {}。",before["words"],after["words"],before["spam_total"],after["spam_total"],before["ham_total"],after["ham_total"],changes["added"],changes["removed"],changes["changed"]);
            tx.execute("INSERT INTO maintainer_actions(actor_id,actor_name,chat_id,command,summary,undo_data,created_at) VALUES (?1,'項目主持人',NULL,'/ml_retrain',?2,?3,?4)",params![actor,summary,serde_json::to_string(&undo)?,Utc::now().to_rfc3339()])?;
            let action_id=tx.last_insert_rowid();
            let result=json!({"request_id":request.request_id,"action_id":action_id,"revision":after_revision,"after":prepared.preview["after"],"updated_at":Utc::now().timestamp()});
            tx.execute("INSERT INTO model_rebuild_requests(request_id,actor_id,payload,result,backup_json,after_revision,action_id) VALUES (?1,?2,?3,?4,?5,?6,?7)",params![request.request_id,actor,serde_json::to_string(&request)?,result.to_string(),serde_json::to_string(&prepared.backup)?,after_revision,action_id])?;
            Ok(Outcome::Saved(result))
        }).await
    }

    pub(super) async fn restore_model_rebuild(&self, request_id: String) -> Result<bool> {
        self.with_model_transaction(move|tx| {
            let saved:Option<(String,String,bool)>=tx.query_row("SELECT backup_json,after_revision,restored FROM model_rebuild_requests WHERE request_id=?1",[&request_id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
            let Some((backup,after,restored))=saved else{return Ok(false);};
            if restored {return Ok(true);}
            if revision(tx)?!=after {return Ok(false);}
            let backup:Backup=serde_json::from_str(&backup)?;
            replace_words(tx,&backup.words)?;
            for (key,value) in backup.meta {match value {
                Some(value)=>{tx.execute("INSERT INTO model_meta(key,value) VALUES (?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",params![key,value])?;},
                None=>{tx.execute("DELETE FROM model_meta WHERE key=?1",[key])?;}
            }}
            tx.execute("UPDATE model_rebuild_requests SET restored=1 WHERE request_id=?1",[request_id])?;
            Ok(true)
        }).await
    }
}
