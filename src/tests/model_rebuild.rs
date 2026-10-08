use super::*;
use crate::model_rebuild::{Outcome, Patch};
use serde_json::{json, Value};

async fn preview(runtime: &Runtime) -> Value {
    for _ in 0..1000 {
        match runtime.preview_model_rebuild(HOST_ID).await.unwrap() {
            Outcome::Saved(v) => return v,
            Outcome::Busy => tokio::time::sleep(Duration::from_millis(5)).await,
            _ => panic!("preview failed"),
        }
    }
    panic!("preview busy")
}
async fn save(runtime: &Runtime, request: Patch) -> Outcome {
    for _ in 0..1000 {
        match runtime
            .save_model_rebuild(HOST_ID, request.clone())
            .await
            .unwrap()
        {
            Outcome::Busy => tokio::time::sleep(Duration::from_millis(5)).await,
            v => return v,
        }
    }
    panic!("save busy")
}
async fn patch(runtime: &Runtime) -> Patch {
    Patch {
        request_id: Uuid::new_v4().to_string(),
        expected_revision: preview(runtime).await["revision"].as_str().unwrap().into(),
    }
}
async fn biased() -> Runtime {
    let runtime = test_runtime().await;
    runtime
        .train_atomic("spam", "casino offer", None)
        .await
        .unwrap();
    runtime
        .train_atomic("ham", "hello friends", None)
        .await
        .unwrap();
    runtime
        .with_conn(|conn| {
            conn.execute(
                "INSERT INTO word_frequencies(word,spam_count,ham_count) VALUES ('manual',99,0)",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    runtime.rebuild_model().await.unwrap();
    runtime
}

#[test]
fn model_commands_do_not_report_success_when_storage_fails() {
    // The command dispatcher has a large future in Windows debug builds.
    std::thread::Builder::new().stack_size(16 * 1024 * 1024).spawn(|| {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
    for command in ["/ml_rebuild", "/ml_retrain"] {
        let runtime = Arc::new(biased().await);
        let before = serde_json::to_value(runtime.model.lock().await.clone()).unwrap();
        runtime.with_conn(move|conn|{conn.execute_batch(if command=="/ml_rebuild"{"DROP TABLE word_frequencies;"}else{"CREATE TRIGGER stop_retrain BEFORE DELETE ON word_frequencies BEGIN SELECT RAISE(ABORT,'write failed'); END;"})?;Ok(())}).await.unwrap();
        let bot = TelegramStub::new(vec![]);
        handle_command(
            bot.bot.clone(),
            runtime.clone(),
            spam_ban_message(HOST_ID, command, None),
        )
        .await
        .unwrap();
        let requests = bot.requests.lock().unwrap().clone();
        assert!(requests.iter().any(|(method, args)| method == "sendmessage"
            && args["text"]
                .as_str()
                .is_some_and(|text| text.contains("失敗"))));
        assert!(!requests.iter().any(|(_, args)| args["text"]
            .as_str()
            .is_some_and(|text| text.contains("已重新載入") || text.contains("已依目前"))));
        assert_eq!(
            serde_json::to_value(runtime.model.lock().await.clone()).unwrap(),
            before
        );
        runtime
            .with_conn(|conn| {
                assert_eq!(
                    conn.query_row("SELECT COUNT(*) FROM maintainer_actions", [], |r| r
                        .get::<_, i64>(0))?,
                    0
                );
                Ok(())
            })
            .await
            .unwrap();
    }
        });
    }).unwrap().join().unwrap();
}

#[tokio::test]
async fn rebuild_preview_is_read_only_and_replays_survive_restart_and_restore() {
    let runtime = biased().await;
    let before = serde_json::to_value(runtime.model.lock().await.clone()).unwrap();
    let changes = runtime
        .with_conn(|conn| Ok(conn.total_changes()))
        .await
        .unwrap();
    let p = preview(&runtime).await;
    assert_eq!(p["changes"]["removed"], 1);
    assert_eq!(p["changes"]["added"], 0);
    assert_eq!(p["sample_rows"], 2);
    assert_eq!(
        runtime
            .with_conn(|conn| Ok(conn.total_changes()))
            .await
            .unwrap(),
        changes
    );
    let request = Patch {
        request_id: Uuid::new_v4().to_string(),
        expected_revision: p["revision"].as_str().unwrap().into(),
    };
    let Outcome::Saved(result) = save(&runtime, request.clone()).await else {
        panic!("save failed")
    };
    assert!(!runtime
        .model
        .lock()
        .await
        .spam_tokens
        .contains_key("manual"));
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let Outcome::Saved(replayed) = save(&restarted, request.clone()).await else {
        panic!("replay failed")
    };
    assert_eq!(replayed, result);
    assert!(restarted
        .restore_model_rebuild(request.request_id.clone())
        .await
        .unwrap());
    assert_eq!(
        serde_json::to_value(restarted.model.lock().await.clone()).unwrap(),
        before
    );
    assert!(matches!(
        save(&restarted, request.clone()).await,
        Outcome::Saved(_)
    ));
    assert_eq!(restarted.model.lock().await.spam_tokens["manual"], 99);
    restarted
        .train_atomic("spam", "new words", None)
        .await
        .unwrap();
    assert!(restarted
        .restore_model_rebuild(request.request_id.clone())
        .await
        .unwrap());
    assert_eq!(restarted.model.lock().await.spam_docs, 2);
    restarted
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM model_rebuild_requests", [], |r| r
                    .get::<_, i64>(0))?,
                1
            );
            assert_eq!(
                conn.query_row(
                    "SELECT COUNT(*) FROM maintainer_actions WHERE command='/ml_retrain'",
                    [],
                    |r| r.get::<_, i64>(0)
                )?,
                1
            );
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM training_samples", [], |r| r
                    .get::<_, i64>(0))?,
                3
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn model_rebuild_conflicts_cover_training_biases_thresholds_and_code_changes() {
    let runtime = biased().await;
    let stale = patch(&runtime).await;
    runtime
        .train_atomic("ham", "fresh sample", None)
        .await
        .unwrap();
    assert!(matches!(save(&runtime, stale).await, Outcome::Conflict));
    let stale = patch(&runtime).await;
    runtime
        .with_conn(|conn| {
            conn.execute(
                "UPDATE word_frequencies SET spam_count=100 WHERE word='manual'",
                [],
            )?;
            conn.execute(
                "UPDATE word_frequencies SET spam_count=99 WHERE word='manual'",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(matches!(save(&runtime, stale).await, Outcome::Conflict));
    let stale = patch(&runtime).await;
    runtime.set_threshold(0.93).await.unwrap();
    assert!(matches!(save(&runtime, stale).await, Outcome::Conflict));
    let mut wrong = patch(&runtime).await;
    wrong.expected_revision = format!("other:{}", wrong.expected_revision);
    assert!(matches!(save(&runtime, wrong).await, Outcome::Conflict));
    let request = patch(&runtime).await;
    assert!(matches!(
        runtime
            .save_model_rebuild(200, request.clone())
            .await
            .unwrap(),
        Outcome::Forbidden
    ));
    assert!(matches!(
        save(&runtime, request.clone()).await,
        Outcome::Saved(_)
    ));
    let mut reused = request.clone();
    reused.expected_revision.push('0');
    assert!(matches!(save(&runtime, reused).await, Outcome::Conflict));
    runtime
        .train_atomic("spam", "later training", None)
        .await
        .unwrap();
    assert!(!runtime
        .restore_model_rebuild(request.request_id)
        .await
        .unwrap());
    assert_eq!(runtime.model.lock().await.spam_docs, 2);
}

#[tokio::test]
async fn failed_rebuild_receipts_roll_back_words_audit_and_revision() {
    let runtime = biased().await;
    let request = patch(&runtime).await;
    let before = serde_json::to_value(runtime.model.lock().await.clone()).unwrap();
    runtime.with_conn(|conn|{conn.execute_batch("CREATE TRIGGER fail_rebuild BEFORE INSERT ON model_rebuild_requests BEGIN SELECT RAISE(ABORT,'receipt failed'); END;")?;Ok(())}).await.unwrap();
    loop {
        match runtime.save_model_rebuild(HOST_ID, request.clone()).await {
            Ok(Outcome::Busy) => tokio::time::sleep(Duration::from_millis(5)).await,
            Err(_) => break,
            _ => panic!("write should fail"),
        }
    }
    assert_eq!(
        preview(&runtime).await["revision"],
        request.expected_revision
    );
    assert_eq!(
        serde_json::to_value(runtime.model.lock().await.clone()).unwrap(),
        before
    );
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row(
                    "SELECT COUNT(*) FROM maintainer_actions WHERE command='/ml_retrain'",
                    [],
                    |r| r.get::<_, i64>(0)
                )?,
                0
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn competing_rebuilds_commit_only_one_receipt_and_audit_entry() {
    let runtime = biased().await;
    let one = patch(&runtime).await;
    let mut two = one.clone();
    two.request_id = Uuid::new_v4().to_string();
    let (a, b) = tokio::join!(save(&runtime, one), save(&runtime, two));
    assert!(matches!(
        (a, b),
        (Outcome::Saved(_), Outcome::Conflict) | (Outcome::Conflict, Outcome::Saved(_))
    ));
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM model_rebuild_requests", [], |r| r
                    .get::<_, i64>(0))?,
                1
            );
            assert_eq!(
                conn.query_row(
                    "SELECT COUNT(*) FROM maintainer_actions WHERE command='/ml_retrain'",
                    [],
                    |r| r.get::<_, i64>(0)
                )?,
                1
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn rebuild_limits_do_not_write_and_migration_keeps_existing_data() {
    let runtime = biased().await;
    let request = patch(&runtime).await;
    runtime.with_conn(|conn|{conn.execute("INSERT INTO training_samples(label,text,created_at) VALUES ('spam',?1,'2026-01-01')",["x".repeat(65537)])?;Ok(())}).await.unwrap();
    assert!(matches!(save(&runtime, request).await, Outcome::TooLarge));
    assert_eq!(runtime.model.lock().await.spam_tokens["manual"], 99);
    runtime.with_conn(|conn|{
        for table in ["training_samples","word_frequencies"] {for op in ["INSERT","UPDATE","DELETE"]{conn.execute_batch(&format!("DROP TRIGGER model_revision_{table}_{op};"))?;}}
        conn.execute_batch("DROP TRIGGER model_revision_meta_INSERT;DROP TRIGGER model_revision_meta_UPDATE;DROP TRIGGER model_revision_meta_DELETE;DROP TABLE model_revision;DROP TABLE model_rebuild_requests;PRAGMA user_version=34;")?;Ok(())
    }).await.unwrap();
    let result = crate::maintenance::check_upgrade(
        &runtime.config.sqlite_path,
        &runtime.config.data_dir.join("model-upgrade"),
    )
    .unwrap();
    let result = serde_json::to_value(result).unwrap();
    assert_eq!(result["schema_before"], 34);
    assert_eq!(result["schema_after"], 35);
    assert_eq!(result["restore"], "ok");
    assert!(serde_json::from_value::<Patch>(
        json!({"request_id":"x","expected_revision":"y","actor_id":HOST_ID})
    )
    .is_err());
}
