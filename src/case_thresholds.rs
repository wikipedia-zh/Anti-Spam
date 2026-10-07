use super::*;
use serde_json::{json, Value};

pub(super) fn record(
    conn: &Connection,
    case_id: &str,
    phase: &str,
    threshold: f64,
    score: f64,
) -> Result<()> {
    let passed = passes_threshold(score, threshold);
    let threshold =
        (threshold.is_finite() && (0.0..=1.0).contains(&threshold)).then_some(threshold);
    conn.execute(
        "INSERT INTO case_threshold_checks(case_id,phase,threshold,passed,checked_at) VALUES (?1,?2,?3,?4,?5)
         ON CONFLICT(case_id,phase) DO UPDATE SET threshold=excluded.threshold,passed=excluded.passed,checked_at=excluded.checked_at
         WHERE phase!='detection' AND (threshold IS NOT excluded.threshold OR passed!=excluded.passed)",
        params![case_id,phase,threshold,passed,Utc::now().to_rfc3339()],
    )?;
    Ok(())
}

pub(super) fn load(conn: &Connection, id: &str) -> Result<Vec<Value>> {
    let mut stmt=conn.prepare("SELECT phase,threshold,passed,checked_at FROM case_threshold_checks WHERE case_id=?1 ORDER BY CASE phase WHEN 'detection' THEN 0 WHEN 'enforcement' THEN 1 ELSE 2 END")?;
    let rows=stmt.query_map([id],|r|Ok(json!({"phase":r.get::<_,String>(0)?,"threshold":r.get::<_,Option<f64>>(1)?,"passed":r.get::<_,bool>(2)?,"checked_at":r.get::<_,String>(3)?})))?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

impl Runtime {
    pub(super) fn migrate_v32_to_v33(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS case_threshold_checks(
            case_id TEXT NOT NULL,
            phase TEXT NOT NULL CHECK(phase IN ('detection','enforcement','network')),
            threshold REAL CHECK(threshold>=0 AND threshold<=1),
            passed INTEGER NOT NULL CHECK(passed IN (0,1)),
            checked_at TEXT NOT NULL,
            PRIMARY KEY(case_id,phase));
            PRAGMA user_version=33;",
        )?;
        tx.commit()?;
        Ok(())
    }
}
