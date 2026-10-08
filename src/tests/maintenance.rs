use super::*;

#[tokio::test]
async fn upgrade_check_preserves_uncheckpointed_data_and_restores_the_old_schema() {
    let runtime = test_runtime().await;
    train_spam(&runtime, "casino gambling", Some("case"))
        .await
        .unwrap();
    runtime
        .set_ot_template(-100, Some("{user} custom {count}"))
        .await
        .unwrap();
    runtime.with_conn(|conn| {
        conn.execute_batch("PRAGMA user_version=24;
            DROP TABLE restriction_jobs; DROP TABLE report_deliveries; DROP TABLE warning_requests; DROP TABLE warning_removals; DROP TABLE rule_captures; DROP TABLE rule_notice_jobs;
            DROP TRIGGER maintainers_revision_INSERT; DROP TRIGGER maintainers_revision_UPDATE; DROP TRIGGER maintainers_revision_DELETE;
            DROP TRIGGER reviewers_revision_INSERT; DROP TRIGGER reviewers_revision_UPDATE; DROP TRIGGER reviewers_revision_DELETE;
            DROP TABLE host_case_requests; DROP TABLE host_role_requests; DROP TABLE role_revision;
            DROP TRIGGER group_text_revision_insert; DROP TRIGGER group_text_revision_update; DROP TRIGGER group_text_revision_delete;")?;
        Ok(())
    }).await.unwrap();
    let writer = Connection::open(&runtime.config.sqlite_path).unwrap();
    writer
        .execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0;
        INSERT INTO model_meta(key,value) VALUES ('uncheckpointed','preserve me');",
        )
        .unwrap();
    let output = runtime.config.data_dir.join("upgrade-check");
    let report = crate::maintenance::check_upgrade(&runtime.config.sqlite_path, &output).unwrap();
    let json = serde_json::to_value(report).unwrap();
    assert_eq!(json["schema_before"], 24);
    assert_eq!(json["schema_after"], 35);
    assert_eq!(json["restore"], "ok");
    assert_eq!(
        writer
            .query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        24
    );
    for name in ["snapshot.db", "upgrade.db", "restored.db"] {
        let db = Connection::open(output.join(name)).unwrap();
        assert_eq!(
            db.query_row(
                "SELECT value FROM model_meta WHERE key='uncheckpointed'",
                [],
                |r| r.get::<_, String>(0)
            )
            .unwrap(),
            "preserve me"
        );
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM training_samples", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
        assert_eq!(
            db.query_row("SELECT ot_template FROM group_warn_settings", [], |r| {
                r.get::<_, String>(0)
            })
            .unwrap(),
            "{user} custom {count}"
        );
    }
    assert!(crate::maintenance::check_upgrade(&runtime.config.sqlite_path, &output).is_err());
}

#[test]
fn upgrade_check_does_not_create_a_missing_source_database() {
    let root = std::env::temp_dir().join(format!("missing-spb-{}", Uuid::new_v4()));
    assert!(
        crate::maintenance::check_upgrade(&root.join("source.db"), &root.join("trial")).is_err()
    );
    assert!(!root.exists());
}
