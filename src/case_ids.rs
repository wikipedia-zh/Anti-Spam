use super::*;
use rusqlite::OptionalExtension;

pub(super) const AMBIGUOUS: &str = "案件短碼對應多筆記錄，請從管理面板複製完整案件 ID。";

#[derive(Debug)]
pub(super) struct Ambiguous;
impl std::fmt::Display for Ambiguous {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(AMBIGUOUS)
    }
}
impl std::error::Error for Ambiguous {}

pub(super) fn display(id: &str) -> String {
    match Uuid::parse_str(id) {
        Ok(uuid) => format!("#{}", &uuid.simple().to_string()[..8]),
        Err(_) => id.to_string(),
    }
}

pub(super) fn prefix(reference: &str) -> Option<String> {
    let value = reference
        .trim()
        .strip_prefix('#')
        .unwrap_or(reference.trim());
    ((8..=32).contains(&value.len()) && value.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| value.to_ascii_lowercase())
}

// Internal keys, callbacks and stored requests continue to use the full ID.
// Resolve only at the boundary where a person supplies a reference.
pub(super) fn resolve(conn: &Connection, reference: &str) -> Result<Option<String>> {
    let reference = reference.trim();
    if let Some(id) = conn
        .query_row("SELECT id FROM cases WHERE id=?1", [reference], |r| {
            r.get(0)
        })
        .optional()?
    {
        return Ok(Some(id));
    }
    let bare = reference.strip_prefix('#').unwrap_or(reference);
    if let Ok(uuid) = Uuid::parse_str(bare) {
        let canonical = uuid.to_string();
        if let Some(id) = conn
            .query_row("SELECT id FROM cases WHERE id=?1", [&canonical], |r| {
                r.get(0)
            })
            .optional()?
        {
            return Ok(Some(id));
        }
    }
    let Some(prefix) = prefix(reference) else {
        return Ok(None);
    };
    let mut statement = conn.prepare("SELECT id FROM cases WHERE length(replace(id,'-',''))=32 AND lower(replace(id,'-','')) LIKE ?1 ORDER BY id LIMIT 2")?;
    let ids = statement
        .query_map([format!("{prefix}%")], |r| r.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    match ids.len() {
        0 => Ok(None),
        1 => Ok(ids.into_iter().next()),
        _ => Err(Ambiguous.into()),
    }
}

impl Runtime {
    pub(super) async fn resolve_case_reference(&self, reference: &str) -> Result<Option<String>> {
        let reference = reference.to_string();
        self.with_conn(move |conn| resolve(conn, &reference)).await
    }

    pub(super) async fn load_case_reference(&self, reference: &str) -> Result<Option<CaseRecord>> {
        match self.resolve_case_reference(reference).await? {
            Some(id) => self.load_case(&id).await,
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_ids_and_short_references_resolve_without_rewriting_records() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE cases(id TEXT PRIMARY KEY);
            INSERT INTO cases VALUES ('12345678-1111-4111-8111-111111111111'),('legacy-case');",
        )
        .unwrap();
        let full = "12345678-1111-4111-8111-111111111111";
        assert_eq!(display(full), "#12345678");
        for reference in [
            full,
            "#12345678",
            "12345678",
            "12345678111141118111111111111111",
        ] {
            assert_eq!(resolve(&conn, reference).unwrap().as_deref(), Some(full));
        }
        assert_eq!(
            resolve(&conn, "legacy-case").unwrap().as_deref(),
            Some("legacy-case")
        );
        for invalid in ["12345", "#1234567_", "%", "#12345678%", "#éééééééé"] {
            assert!(resolve(&conn, invalid).unwrap().is_none());
        }
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM cases", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            2
        );
    }

    #[test]
    fn collisions_require_a_longer_reference_and_never_select_the_first_case() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE cases(id TEXT PRIMARY KEY);
            INSERT INTO cases VALUES ('abcdef12-1111-4111-8111-111111111111'),('abcdef12-2222-4222-8222-222222222222');").unwrap();
        assert!(resolve(&conn, "#ABCDEF12")
            .unwrap_err()
            .downcast_ref::<Ambiguous>()
            .is_some());
        assert_eq!(
            resolve(&conn, "#abcdef121").unwrap().as_deref(),
            Some("abcdef12-1111-4111-8111-111111111111")
        );
        assert_eq!(
            resolve(&conn, "abcdef12-2222-4222-8222-222222222222")
                .unwrap()
                .as_deref(),
            Some("abcdef12-2222-4222-8222-222222222222")
        );
    }
}
