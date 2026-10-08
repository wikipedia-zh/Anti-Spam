use super::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

static MODEL_WORK: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);
const MAX_ROWS: i64 = 20_000;
const MAX_BYTES: i64 = 16 * 1024 * 1024;
const MAX_SAMPLE_BYTES: i64 = 64 * 1024;

#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Request {
    Summary {},
    Score { text: String },
    Evaluate {},
}

pub(super) enum Outcome {
    Ready(Value),
    Invalid,
    Forbidden,
    Busy,
}

fn threshold(value: f64) -> Option<f64> {
    (value.is_finite() && (0.0..=1.0).contains(&value)).then_some(value)
}

impl Runtime {
    pub(super) async fn host_model(&self, actor: i64, request: Request) -> Result<Outcome> {
        if !is_host(actor) {
            return Ok(Outcome::Forbidden);
        }
        if matches!(&request, Request::Score { text } if text.chars().count()>4000) {
            return Ok(Outcome::Invalid);
        }
        let Ok(permit) = MODEL_WORK.try_acquire() else {
            return Ok(Outcome::Busy);
        };
        let default = self.config.spam_threshold;
        match request {
            Request::Summary {} => {
                let model = self.model.clone().lock_owned().await;
                self.with_conn(move |conn| {
                    let _permit = permit;
                    let tx = conn.transaction()?;
                    let live = threshold(Self::load_threshold(&tx)?.unwrap_or(default));
                    let (rows,spam,ham,invalid,newest):(i64,i64,i64,i64,Option<String>)=tx.query_row(
                        "SELECT COUNT(*),COALESCE(SUM(label='spam'),0),COALESCE(SUM(label='ham'),0),COALESCE(SUM(label NOT IN ('spam','ham')),0),MAX(created_at) FROM training_samples",[],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?)))?;
                    let overrides:i64=tx.query_row("SELECT COUNT(*) FROM group_module_settings WHERE spam_threshold_override IS NOT NULL AND chat_id NOT IN (SELECT chat_id FROM banned_groups)",[],|r|r.get(0))?;
                    let now=Utc::now();
                    let since=now-chrono::Duration::days(30);
                    let (approved,rejected):(i64,i64)=tx.query_row("SELECT COALESCE(SUM(decision='approve'),0),COALESCE(SUM(decision='reject'),0) FROM training_reviews WHERE julianday(decided_at)>=julianday(?1) AND julianday(decided_at)<=julianday(?2)",params![since.to_rfc3339(),now.to_rfc3339()],|r|Ok((r.get(0)?,r.get(1)?)))?;
                    let result=json!({"updated_at":Utc::now().timestamp(),"global_threshold":live,
                        "samples":{"rows":rows,"spam":spam,"ham":ham,"invalid":invalid,"newest":newest},
                        "model":{"spam_docs":model.spam_docs,"ham_docs":model.ham_docs,"spam_vocabulary":model.spam_tokens.len(),"ham_vocabulary":model.ham_tokens.len()},
                        "group_overrides":overrides,"training_reviews":{"since":since.timestamp(),"approved":approved,"rejected":rejected}});
                    tx.commit()?;
                    Ok(Outcome::Ready(result))
                }).await
            }
            Request::Score { text } => {
                let model = self.model.clone().lock_owned().await;
                let (model, live, permit, now) = self
                    .with_conn(move |conn| {
                        Ok((
                            model,
                            threshold(Self::load_threshold(conn)?.unwrap_or(default)),
                            permit,
                            Utc::now().timestamp(),
                        ))
                    })
                    .await?;
                tokio::task::spawn_blocking(move || {
                    let _permit = permit;
                    let mut report = score_debug_from_text(&model, &text);
                    let count = report.tokens.len();
                    report.tokens.sort_by(|a,b|b.delta.abs().total_cmp(&a.delta.abs()).then(a.token.cmp(&b.token)));
                    let tokens:Vec<_>=report.tokens.into_iter().take(80).map(|t|json!({"token":t.token,"spam_count":t.spam_count,"ham_count":t.ham_count,"delta":t.delta})).collect();
                    Ok(Outcome::Ready(json!({"updated_at":now,"score":report.score,"global_threshold":live,
                        "passes_global":live.map(|t|passes_threshold(report.score,t)),"token_count":count,"tokens":tokens,
                        "model":{"spam_docs":model.spam_docs,"ham_docs":model.ham_docs}})))
                }).await?
            }
            Request::Evaluate {} => {
                // Copy one database snapshot, then release SQLite before tokenizing.
                let (samples,live,now,size,permit)=self.with_conn(move |conn| {
                    let tx=conn.transaction()?;
                    let live=threshold(Self::load_threshold(&tx)?.unwrap_or(default));
                    let size:(i64,i64,i64)=tx.query_row("SELECT COUNT(*),COALESCE(SUM(length(CAST(text AS BLOB))),0),COALESCE(MAX(length(CAST(text AS BLOB))),0) FROM training_samples",[],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?;
                    let samples=if live.is_some() && size.0<=MAX_ROWS && size.1<=MAX_BYTES && size.2<=MAX_SAMPLE_BYTES {
                        let mut stmt=tx.prepare("SELECT label,text FROM training_samples ORDER BY id")?;
                        let rows=stmt.query_map([],|r|Ok(evaluation::Sample {label:r.get(0)?,text:r.get(1)?}))?;
                        Some(rows.collect::<rusqlite::Result<Vec<_>>>()?)
                    } else {None};
                    let now=Utc::now().timestamp();
                    tx.commit()?;
                    Ok((samples,live,now,size,permit))
                }).await?;
                tokio::task::spawn_blocking(move || {
                    let _permit=permit;
                    let Some(samples)=samples else {
                        return Ok(Outcome::Ready(json!({"status":if live.is_none(){"invalid_threshold"}else{"too_large"},"updated_at":now,
                            "rows":size.0,"bytes":size.1,"max_rows":MAX_ROWS,"max_bytes":MAX_BYTES,"max_sample_bytes":MAX_SAMPLE_BYTES})));
                    };
                    let mut hash=Sha256::new();
                    for sample in &samples {
                        for s in [&sample.label,&sample.text] { hash.update((s.len() as u64).to_le_bytes());hash.update(s.as_bytes()); }
                    }
                    let report=evaluation::evaluate(&samples,0.2,live.unwrap())?;
                    Ok(Outcome::Ready(json!({"status":"ready","updated_at":now,"sample_fingerprint":format!("{:x}",hash.finalize()),"report":report})))
                }).await?
            }
        }
    }
}
