use super::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Read;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileProof {
    path: String,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Report {
    backup_at: i64,
    checked_at: i64,
    backup: FileProof,
    upgraded: FileProof,
    restored: FileProof,
    schema_before: i64,
    schema_after: i64,
    tables_verified: u64,
    integrity: String,
    restore: String,
}

fn check_file(root: &std::path::Path, proof: &FileProof) -> Result<u64> {
    use std::path::Component;
    let relative = std::path::Path::new(&proof.path);
    anyhow::ensure!(
        !relative.as_os_str().is_empty()
            && relative
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "invalid artifact path"
    );
    let path = root.join(relative).canonicalize()?;
    anyhow::ensure!(
        path.starts_with(root),
        "artifact outside database directory"
    );
    let mut file = std::fs::File::open(path)?;
    let meta = file.metadata()?;
    anyhow::ensure!(
        meta.is_file() && meta.len() <= 1024 * 1024 * 1024 && proof.sha256.len() == 64,
        "invalid artifact"
    );
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut size = 0;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        size += n as u64;
        anyhow::ensure!(size <= meta.len(), "artifact changed");
        hash.update(&buffer[..n]);
    }
    anyhow::ensure!(
        size == meta.len() && format!("{:x}", hash.finalize()) == proof.sha256,
        "artifact changed"
    );
    Ok(size)
}

pub(super) fn read_report(database: &std::path::Path) -> Value {
    let report_path = database.with_extension("maintenance.json");
    let file = match std::fs::File::open(&report_path) {
        Ok(f) => f,
        Err(e) => {
            return json!({"status":if e.kind()==std::io::ErrorKind::NotFound {"missing"}else{"unverified"}})
        }
    };
    let checked = (|| -> Result<Value> {
        let mut bytes = Vec::new();
        file.take(16385).read_to_end(&mut bytes)?;
        anyhow::ensure!(bytes.len() <= 16384, "report too large");
        let report: Report = serde_json::from_slice(&bytes)?;
        anyhow::ensure!(
            report.backup_at > 0
                && report.checked_at >= report.backup_at
                && report.checked_at <= Utc::now().timestamp() + 300
                && report.schema_before >= 17
                && report.schema_after >= report.schema_before
                && report.tables_verified > 0
                && report.integrity == "ok"
                && report.restore == "ok",
            "incomplete report"
        );
        let root = database
            .canonicalize()?
            .parent()
            .context("missing database directory")?
            .to_path_buf();
        let size = check_file(&root, &report.backup)?;
        check_file(&root, &report.upgraded)?;
        check_file(&root, &report.restored)?;
        Ok(
            json!({"status":"verified","backup_at":report.backup_at,"checked_at":report.checked_at,
            "schema_before":report.schema_before,"schema_after":report.schema_after,"tables_verified":report.tables_verified,"backup_bytes":size}),
        )
    })();
    checked.unwrap_or_else(|_| json!({"status":"unverified"}))
}

pub(super) fn snapshot(conn: &Connection, backup: Value, started_at: i64) -> Result<Value> {
    let mut s=conn.prepare(&format!("SELECT kind,COUNT(*),SUM(last_error IS NOT NULL),MIN(next_attempt_at) FROM ({}) GROUP BY kind ORDER BY kind",queue_status::WORK))?;
    let rows=s.query_map([],|r|Ok(json!({"kind":r.get::<_,String>(0)?,"pending":r.get::<_,i64>(1)?,"errors":r.get::<_,i64>(2)?,"next_attempt_at":r.get::<_,i64>(3)?})))?;
    let queues = rows.collect::<rusqlite::Result<Vec<_>>>()?;
    let mut groups = serde_json::Map::new();
    let mut s=conn.prepare("SELECT COALESCE(a.state,'unknown'),COUNT(*) FROM group_module_settings g LEFT JOIN group_access a ON a.chat_id=g.chat_id GROUP BY COALESCE(a.state,'unknown')")?;
    for row in s.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
        let (key, count) = row?;
        groups.insert(key, json!(count));
    }
    Ok(
        json!({"version":env!("GIT_HASH"),"schema":conn.query_row("PRAGMA user_version",[],|r|r.get::<_,i64>(0))?,
        "started_at":started_at,"database":"readable","worker_health":"unknown","backup":backup,"queues":queues,"groups":groups,
        "controls":operations::controls(conn)?,"telegram_not_before":conn.query_row("SELECT not_before FROM telegram_retry_state WHERE id=1",[],|r|r.get::<_,i64>(0))?}),
    )
}
