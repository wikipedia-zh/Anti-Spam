use super::*;
use rusqlite::{types::ValueRef, DatabaseName, OpenFlags};
use sha2::{Digest, Sha256};
use std::path::Path;

#[derive(Serialize)]
pub(super) struct UpgradeCheck {
    schema_before: i64,
    schema_after: i64,
    tables_verified: usize,
    integrity: &'static str,
    restore: &'static str,
}

fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn fingerprint(conn: &Connection, table: &str, columns: &[String]) -> Result<Vec<u8>> {
    let sql = format!(
        "SELECT {} FROM {} ORDER BY rowid",
        columns
            .iter()
            .map(|c| quote(c))
            .collect::<Vec<_>>()
            .join(","),
        quote(table)
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut hash = Sha256::new();
    while let Some(row) = rows.next()? {
        hash.update([0xff]);
        for index in 0..columns.len() {
            match row.get_ref(index)? {
                ValueRef::Null => hash.update([0]),
                ValueRef::Integer(v) => {
                    hash.update([1]);
                    hash.update(v.to_le_bytes());
                }
                ValueRef::Real(v) => {
                    hash.update([2]);
                    hash.update(v.to_bits().to_le_bytes());
                }
                ValueRef::Text(v) | ValueRef::Blob(v) => {
                    hash.update([if matches!(row.get_ref(index)?, ValueRef::Text(_)) {
                        3
                    } else {
                        4
                    }]);
                    hash.update((v.len() as u64).to_le_bytes());
                    hash.update(v);
                }
            }
        }
    }
    Ok(hash.finalize().to_vec())
}

fn readonly(path: &Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    conn.busy_timeout(Duration::from_secs(5))?;
    Ok(conn)
}

fn integrity(conn: &Connection) -> Result<()> {
    let result: String = conn.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
    anyhow::ensure!(result == "ok", "database integrity check failed");
    Ok(())
}

/// Work on SQLite backups in a new private directory. No bot configuration
/// or Telegram connection is needed, and the source is opened read-only.
pub(super) fn check_upgrade(source: &Path, output: &Path) -> Result<UpgradeCheck> {
    let source = readonly(source)?;
    let directory = std::fs::DirBuilder::new();
    #[cfg(unix)]
    let mut directory = directory;
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        directory.mode(0o700);
    }
    directory
        .create(output)
        .context("use a new directory for the upgrade check")?;
    source.backup(DatabaseName::Main, output.join("snapshot.db"), None)?;
    drop(source);
    let snapshot = readonly(&output.join("snapshot.db"))?;
    integrity(&snapshot)?;
    let before: i64 = snapshot.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    anyhow::ensure!(
        before >= 17,
        "upgrade check expects an SPB database at schema 17 or later"
    );
    let mut target = Connection::open_in_memory()?;
    Runtime::init_db(&mut target)?;
    let target_version = target.query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0))?;
    anyhow::ensure!(
        before <= target_version,
        "database is newer than this binary"
    );
    let tables = {
        let mut stmt = snapshot.prepare("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    let mut expected = Vec::new();
    for table in tables {
        let mut stmt = snapshot.prepare(&format!("PRAGMA table_info({})", quote(&table)))?;
        let columns = stmt
            .query_map([], |r| r.get::<_, String>(1))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let hash = fingerprint(&snapshot, &table, &columns)?;
        expected.push((table, columns, hash));
    }
    snapshot.backup(DatabaseName::Main, output.join("upgrade.db"), None)?;
    let mut upgraded = Connection::open(output.join("upgrade.db"))?;
    Runtime::init_db(&mut upgraded)?;
    integrity(&upgraded)?;
    let after = upgraded.query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0))?;
    anyhow::ensure!(
        after == target_version,
        "migration did not reach the current schema"
    );
    for (table, columns, hash) in &expected {
        anyhow::ensure!(
            fingerprint(&upgraded, table, columns)? == *hash,
            "migration changed existing data in {table}"
        );
    }
    Runtime::load_model(&upgraded)?;
    upgraded.backup(DatabaseName::Main, output.join("restored.db"), None)?;
    let mut restored = Connection::open(output.join("restored.db"))?;
    restored.restore(
        DatabaseName::Main,
        output.join("snapshot.db"),
        None::<fn(rusqlite::backup::Progress)>,
    )?;
    integrity(&restored)?;
    anyhow::ensure!(
        restored.query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0))? == before,
        "restore schema mismatch"
    );
    for (table, columns, hash) in &expected {
        anyhow::ensure!(
            fingerprint(&restored, table, columns)? == *hash,
            "restore mismatch in {table}"
        );
    }
    let restored_tables: i64 = restored.query_row(
        "SELECT COUNT(*) FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'",
        [],
        |r| r.get(0),
    )?;
    anyhow::ensure!(
        restored_tables == expected.len() as i64,
        "restore retained unexpected tables"
    );
    Ok(UpgradeCheck {
        schema_before: before,
        schema_after: after,
        tables_verified: expected.len(),
        integrity: "ok",
        restore: "ok",
    })
}
