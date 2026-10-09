use super::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn proof(root: &std::path::Path, path: &std::path::Path) -> Value {
    json!({"path":path.strip_prefix(root).unwrap().to_string_lossy(),"sha256":format!("{:x}",Sha256::digest(std::fs::read(path).unwrap()))})
}

#[test]
fn maintenance_report_requires_matching_backup_upgrade_and_restore_files() {
    let root = std::env::temp_dir().join(format!("spb-maintenance-report-{}", Uuid::new_v4()));
    std::fs::create_dir(&root).unwrap();
    let db = root.join("bot.db");
    let mut conn = Connection::open(&db).unwrap();
    Runtime::init_db(&mut conn).unwrap();
    drop(conn);
    assert_eq!(
        crate::host_maintenance::read_report(&db)["status"],
        "missing"
    );
    let check = root.join("check");
    let result =
        serde_json::to_value(crate::maintenance::check_upgrade(&db, &check).unwrap()).unwrap();
    let mut record = json!({"backup_at":Utc::now().timestamp(),"checked_at":Utc::now().timestamp(),
        "backup":proof(&root,&check.join("snapshot.db")),"upgraded":proof(&root,&check.join("upgrade.db")),"restored":proof(&root,&check.join("restored.db")),
        "schema_before":result["schema_before"],"schema_after":result["schema_after"],"tables_verified":result["tables_verified"],"integrity":"ok","restore":"ok"});
    let manifest = db.with_extension("maintenance.json");
    std::fs::write(&manifest, record.to_string()).unwrap();
    let v = crate::host_maintenance::read_report(&db);
    assert_eq!(v["status"], "verified");
    assert_eq!(v["schema_after"], 41);
    assert!(v.get("backup").is_none());
    let correct = record.clone();
    for invalid in [
        json!({"path":"../outside.db","sha256":"0".repeat(64)}),
        json!({"path":db.to_string_lossy(),"sha256":"0".repeat(64)}),
        json!({"path":"check/snapshot.db","sha256":"0".repeat(64)}),
    ] {
        record["backup"] = invalid;
        std::fs::write(&manifest, record.to_string()).unwrap();
        assert_eq!(
            crate::host_maintenance::read_report(&db)["status"],
            "unverified"
        );
    }
    std::fs::write(&manifest, correct.to_string()).unwrap();
    std::fs::write(check.join("restored.db"), b"changed").unwrap();
    assert_eq!(
        crate::host_maintenance::read_report(&db)["status"],
        "unverified"
    );
    std::fs::write(&manifest, "x".repeat(16385)).unwrap();
    assert_eq!(
        crate::host_maintenance::read_report(&db)["status"],
        "unverified"
    );
}

#[tokio::test]
async fn maintenance_snapshot_does_not_infer_worker_health_from_a_database_read() {
    let runtime = test_runtime().await;
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    let obs = runtime.group_access_revision(-300).await.unwrap();
    runtime
        .record_group_access_check(-300, obs, "left")
        .await
        .unwrap();
    let page = runtime
        .host_query(crate::host_panel::Query {
            view: "maintenance".into(),
            search: String::new(),
            filter: String::new(),
            offset: 0,
            created_from: None,
            created_before: None,
        })
        .await
        .unwrap();
    let v = &page["items"][0];
    assert_eq!(v["database"], "readable");
    assert_eq!(v["worker_health"], "unknown");
    assert_eq!(v["backup"]["status"], "missing");
    assert_eq!(v["groups"]["left"], 1);
    assert_eq!(v["schema"], 41);
}
