use super::*;
use rusqlite::OpenFlags;
use std::path::Path;

pub(super) async fn handle(bot: &Bot, message: &Message, reference: &str) -> ResponseResult<()> {
    if !message.chat.is_private()
        || !message
            .from
            .as_ref()
            .is_some_and(|user| is_host(user.id.0 as i64))
    {
        return Ok(());
    }
    let reference = reference.to_owned();
    let text = tokio::task::spawn_blocking(move || -> Result<String> {
        let path = std::env::var("HEXDRILL_STATE_DB")?;
        lookup(Path::new(&path), &reference, Utc::now().timestamp())
    })
    .await
    .ok()
    .and_then(Result::ok)
    .unwrap_or_else(|| "目前無法讀取 SPB-WATCH 紀錄，請稍後重試。".into());
    bot.send_message(message.chat.id, text).await?;
    Ok(())
}

fn lookup(path: &Path, reference: &str, now: i64) -> Result<String> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    conn.busy_timeout(Duration::from_secs(2))?;
    conn.execute_batch("PRAGMA query_only=ON; PRAGMA trusted_schema=OFF;")?;
    let reference = reference.trim().to_ascii_lowercase();
    let (column, value, minimum) = if let Some(id) = reference.strip_prefix("sync:") {
        ("id", id.replace('-', ""), 12)
    } else if let Some(id) = reference.strip_prefix('#') {
        ("case_id", id.replace('-', ""), 8)
    } else if reference.bytes().all(|c| c.is_ascii_digit())
        && reference
            .parse::<i64>()
            .is_ok_and(|id| id > 0 && id < (1_i64 << 52))
    {
        ("user_id", reference, 0)
    } else {
        ("id", reference.replace('-', ""), 12)
    };
    let value = if minimum > 0 {
        if value.len() < minimum
            || value.len() > 32
            || !value.bytes().all(|c| c.is_ascii_hexdigit())
        {
            return Ok("用法：/sync sync:追蹤碼、/sync #案件碼 或 /sync 使用者ID".into());
        }
        let mut stmt = conn.prepare(&format!("SELECT DISTINCT {column} FROM operation_reports WHERE replace({column},'-','') LIKE ?1 LIMIT 2"))?;
        let ids = stmt
            .query_map([format!("{value}%")], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        if ids.len() > 1 {
            return Ok("短碼對應多筆紀錄，請使用完整編號。".into());
        }
        let Some(id) = ids.into_iter().next() else {
            return Ok("找不到同步紀錄；不代表此人沒有 SPB 案件。".into());
        };
        id
    } else {
        value
    };
    let mut stmt = conn.prepare(&format!("SELECT body,updated_at FROM operation_reports WHERE {column}=?1 ORDER BY created_at DESC,id DESC LIMIT 3"))?;
    let rows = stmt
        .query_map([value], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    if rows.is_empty() {
        return Ok("找不到同步紀錄；不代表此人沒有 SPB 案件。".into());
    }
    let mut text = if rows.iter().any(|row| now - row.1 > 90) {
        "⚠️ 資料未能更新，以下為上次讀取結果。\n".to_string()
    } else {
        String::new()
    };
    text.push_str(
        &rows
            .into_iter()
            .map(|row| row.0)
            .collect::<Vec<_>>()
            .join("\n\n"),
    );
    Ok(text.chars().take(3500).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_is_readonly_and_rejects_ambiguous_references() {
        let path = std::env::temp_dir().join(format!("watch-{}.db", uuid::Uuid::new_v4()));
        assert!(lookup(&path, "sync:123456789abc", 100).is_err());
        assert!(!path.exists());
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE operation_reports(id TEXT,user_id INTEGER,case_id TEXT,body TEXT,created_at INTEGER,updated_at INTEGER);
            INSERT INTO operation_reports VALUES ('12345678-9abc-4000-8000-000000000001',42,'abcdef01-0000-4000-8000-000000000001','record one',1,10);
            INSERT INTO operation_reports VALUES ('12345678-9abc-4000-8000-000000000002',42,'abcdef01-0000-4000-8000-000000000001','record two',2,10);").unwrap();
        assert!(lookup(&path, "sync:123456789abc", 100)
            .unwrap()
            .contains("多筆"));
        assert!(
            lookup(&path, "sync:12345678-9abc-4000-8000-000000000001", 101)
                .unwrap()
                .contains("未能更新")
        );
        assert!(lookup(&path, "#abcdef01", 100)
            .unwrap()
            .contains("record two"));
        assert!(lookup(&path, "42", 100).unwrap().contains("record one"));
        assert!(lookup(&path, "' OR 1=1", 100).unwrap().starts_with("用法"));
        drop(conn);
        std::fs::remove_file(path).unwrap();
    }
}
