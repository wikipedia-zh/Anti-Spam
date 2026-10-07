use super::*;
use crate::host_cases::{Outcome, Query, Reverse, Review};
use serde_json::{json, Value};

async fn case(runtime: &Runtime, report: bool) -> CaseRecord {
    let mut case = dummy_case(
        if report {
            ActionKind::PendingReport
        } else {
            ActionKind::SpamBan
        },
        -100,
        200,
        Utc::now(),
    );
    case.status = if report { "pending_review" } else { "done" }.into();
    case.evidence_text = "casino promotional offer".into();
    case.actor_user_id = Some(300);
    runtime.persist_case(&case).await.unwrap();
    case
}
async fn preview(runtime: &Runtime, id: &str) -> Value {
    runtime
        .host_case(Query {
            case_id: id.into(),
            offset: 0,
        })
        .await
        .unwrap()
        .unwrap()
}
async fn request(runtime: &Runtime, case: &CaseRecord, decision: &str) -> Review {
    let value = preview(runtime, &case.id).await;
    assert_eq!(value["review"]["can_review"], true);
    Review {
        request_id: Uuid::new_v4().to_string(),
        case_id: case.id.clone(),
        target_user_id: case.target_user_id,
        expected_revision: value["revision"].as_str().unwrap().into(),
        kind: value["review"]["kind"].as_str().unwrap().into(),
        decision: decision.into(),
    }
}
async fn scalar(runtime: &Runtime, sql: &str) -> i64 {
    let sql = sql.to_string();
    runtime
        .with_conn(move |conn| Ok(conn.query_row(&sql, [], |r| r.get(0))?))
        .await
        .unwrap()
}

#[tokio::test]
async fn report_rejection_replays_without_retraining_or_repeating_strikes_and_retries_notice() {
    let runtime = test_runtime().await;
    let case = case(&runtime, true).await;
    runtime
        .set_report_confirmation(&case.id, -100, 42)
        .await
        .unwrap();
    let patch = request(&runtime, &case, "reject").await;
    assert_eq!(
        preview(&runtime, &case.id).await["review"]["message_known"],
        false
    );
    assert!(matches!(
        runtime
            .review_host_case(HOST_ID, patch.clone())
            .await
            .unwrap(),
        Outcome::Saved(_)
    ));
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    assert!(matches!(
        restarted.review_host_case(HOST_ID, patch).await.unwrap(),
        Outcome::Saved(_)
    ));
    assert_eq!(restarted.model.lock().await.ham_docs, 1);
    assert_eq!(
        scalar(
            &restarted,
            "SELECT rejected_count FROM report_offenses WHERE user_id=300"
        )
        .await,
        1
    );
    assert_eq!(
        scalar(
            &restarted,
            "SELECT COUNT(*) FROM maintainer_actions WHERE command='面板審核'"
        )
        .await,
        1
    );
    let failed = TelegramStub::with_failures(vec![], vec![("editmessagetext".into(), -100)]);
    crate::moderation_queue::deliver_review_updates(&failed.bot, &restarted, None)
        .await
        .unwrap();
    assert_eq!(
        scalar(
            &restarted,
            "SELECT COUNT(*) FROM review_updates WHERE last_error IS NOT NULL"
        )
        .await,
        1
    );
    restarted
        .with_conn(|conn| {
            conn.execute("UPDATE review_updates SET next_attempt_at=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
    let telegram = TelegramStub::new(vec![]);
    crate::moderation_queue::deliver_review_updates(&telegram.bot, &restarted, None)
        .await
        .unwrap();
    let calls = telegram.requests.lock().unwrap();
    let edits: Vec<_> = calls
        .iter()
        .filter(|(method, _)| method == "editmessagetext")
        .collect();
    assert_eq!(edits.len(), 1);
    assert_eq!(edits[0].1["chat_id"], -100);
    assert_eq!(edits[0].1["message_id"], 42);
}

#[tokio::test]
async fn training_approval_is_atomic_with_network_jobs_and_reversal_wins_over_replay() {
    let runtime = test_runtime().await;
    let case = case(&runtime, false).await;
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    let patch = request(&runtime, &case, "approve").await;
    let (first, second) = tokio::join!(
        runtime.review_host_case(HOST_ID, patch.clone()),
        runtime.review_host_case(HOST_ID, patch.clone())
    );
    assert!(matches!(first.unwrap(), Outcome::Saved(_)));
    assert!(matches!(second.unwrap(), Outcome::Saved(_)));
    assert_eq!(runtime.model.lock().await.spam_docs, 1);
    assert_eq!(
        scalar(
            &runtime,
            "SELECT COUNT(*) FROM network_deliveries WHERE state='pending'"
        )
        .await,
        1
    );
    let view = preview(&runtime, &case.id).await;
    assert_eq!(view["review"]["decision"], "approve");
    let reverse = Reverse {
        request_id: Uuid::new_v4().to_string(),
        case_id: case.id.clone(),
        target_user_id: 200,
        expected_revision: view["revision"].as_str().unwrap().into(),
    };
    assert!(matches!(
        runtime.reverse_host_case(HOST_ID, reverse).await.unwrap(),
        Outcome::Saved(_)
    ));
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    assert!(matches!(
        restarted.review_host_case(HOST_ID, patch).await.unwrap(),
        Outcome::Saved(_)
    ));
    assert_eq!(
        scalar(
            &restarted,
            "SELECT COUNT(*) FROM network_deliveries WHERE state='pending'"
        )
        .await,
        0
    );
    assert_eq!(
        scalar(&restarted, "SELECT COUNT(*) FROM training_reviews").await,
        1
    );
    assert_eq!(
        restarted.load_case(&case.id).await.unwrap().unwrap().status,
        "reversal_pending"
    );
}

#[tokio::test]
async fn callback_and_panel_cannot_apply_opposing_report_decisions() {
    let runtime = test_runtime().await;
    let case = case(&runtime, true).await;
    let patch = request(&runtime, &case, "reject").await;
    let callback = async {
        let guard = runtime.review_guard(&case.id).await;
        runtime
            .decide_report(&case, "approve", (HOST_ID, "Host".into()), (-1, 42), guard)
            .await
            .unwrap()
    };
    let (panel, callback) = tokio::join!(runtime.review_host_case(HOST_ID, patch), callback);
    match panel.unwrap() {
        Outcome::Saved(_) => {
            assert!(!callback);
            assert_eq!(runtime.model.lock().await.ham_docs, 1);
            assert_eq!(
                scalar(&runtime, "SELECT COUNT(*) FROM origin_ban_jobs").await,
                0
            );
        }
        Outcome::Conflict => {
            assert!(callback);
            assert_eq!(runtime.model.lock().await.ham_docs, 0);
            assert_eq!(
                scalar(&runtime, "SELECT COUNT(*) FROM origin_ban_jobs").await,
                1
            );
        }
        result => panic!("unexpected {result:?}"),
    }
}

#[tokio::test]
async fn changed_reporter_permissions_and_failed_receipts_do_not_leave_partial_decisions() {
    let runtime = test_runtime().await;
    let case = case(&runtime, true).await;
    let stale = request(&runtime, &case, "reject").await;
    runtime
        .with_conn(|conn| {
            conn.execute(
                "INSERT INTO maintainers(user_id,added_by,created_at) VALUES (300,?1,?2)",
                params![HOST_ID, Utc::now().to_rfc3339()],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(matches!(
        runtime.review_host_case(HOST_ID, stale).await.unwrap(),
        Outcome::Conflict
    ));
    let patch = request(&runtime, &case, "reject").await;
    assert!(matches!(
        runtime.review_host_case(300, patch.clone()).await.unwrap(),
        Outcome::Forbidden
    ));
    runtime.with_conn(|conn|{conn.execute_batch("CREATE TRIGGER fail_review_receipt BEFORE INSERT ON host_case_requests BEGIN SELECT RAISE(ABORT,'receipt failure'); END;")?;Ok(())}).await.unwrap();
    assert!(runtime
        .review_host_case(HOST_ID, patch.clone())
        .await
        .is_err());
    assert_eq!(runtime.model.lock().await.ham_docs, 0);
    assert_eq!(
        scalar(&runtime, "SELECT COUNT(*) FROM training_samples").await,
        0
    );
    assert_eq!(
        scalar(&runtime, "SELECT COUNT(*) FROM maintainer_actions").await,
        0
    );
    assert_eq!(
        runtime.load_case(&case.id).await.unwrap().unwrap().status,
        "pending_review"
    );
    runtime
        .with_conn(|conn| {
            conn.execute_batch("DROP TRIGGER fail_review_receipt;")?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(matches!(
        runtime
            .review_host_case(HOST_ID, patch.clone())
            .await
            .unwrap(),
        Outcome::Saved(_)
    ));
    assert_eq!(
        scalar(&runtime, "SELECT COUNT(*) FROM report_offenses").await,
        0
    );
    let mut reused = patch;
    reused.decision = "approve".into();
    assert!(matches!(
        runtime.review_host_case(HOST_ID, reused).await.unwrap(),
        Outcome::Conflict
    ));
}

#[tokio::test]
async fn training_card_is_saved_once_and_pending_training_disappears_after_rejection() {
    let runtime = test_runtime().await;
    let case = case(&runtime, false).await;
    let telegram = TelegramStub::new(vec![]);
    for _ in 0..2 {
        queue_training_review(
            &telegram.bot,
            &runtime,
            &case,
            runtime.origin_guards(&case).await,
        )
        .await
        .unwrap();
    }
    assert_eq!(
        telegram
            .requests
            .lock()
            .unwrap()
            .iter()
            .filter(|(m, _)| m == "sendmessage")
            .count(),
        1
    );
    assert_eq!(
        preview(&runtime, &case.id).await["review"]["message_known"],
        true
    );
    let query =
        || serde_json::from_value(json!({"view":"cases","filter":"pending_training"})).unwrap();
    assert_eq!(
        runtime.host_query(query()).await.unwrap()["items"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let patch = request(&runtime, &case, "reject").await;
    assert!(matches!(
        runtime.review_host_case(HOST_ID, patch).await.unwrap(),
        Outcome::Saved(_)
    ));
    assert!(runtime.host_query(query()).await.unwrap()["items"]
        .as_array()
        .unwrap()
        .is_empty());
    assert_eq!(runtime.model.lock().await.spam_docs, 0);
    assert_eq!(
        runtime.load_case(&case.id).await.unwrap().unwrap().status,
        "done"
    );
    assert_eq!(
        scalar(&runtime, "SELECT COUNT(*) FROM network_deliveries").await,
        0
    );
    crate::moderation_queue::deliver_review_updates(&telegram.bot, &runtime, None)
        .await
        .unwrap();
    assert!(telegram.requests.lock().unwrap().iter().any(
        |(m, p)| m == "editmessagetext" && p["text"].as_str().unwrap().contains("本群封禁保留")
    ));
}

#[tokio::test]
async fn review_upgrade_preserves_old_notifications_and_recovers_legacy_pending_training() {
    let runtime = test_runtime().await;
    let pending = case(&runtime, false).await;
    let mut empty = dummy_case(ActionKind::SpamBan, -100, 201, Utc::now());
    empty.evidence_text = String::new();
    runtime.persist_case(&empty).await.unwrap();
    runtime.with_conn(|conn| {
        conn.execute_batch("DROP TABLE review_updates; DROP TABLE training_review_locations;")?;
        Runtime::migrate_v23_to_v24(conn)?;
        conn.execute_batch("INSERT INTO review_updates(rowid,case_id,kind,decision,chat_id,message_id,attempts,last_error) VALUES (37,'old','report','reject',-1,42,3,'retry'); PRAGMA user_version=31;")?;
        Ok(())
    }).await.unwrap();
    let result = crate::maintenance::check_upgrade(
        &runtime.config.sqlite_path,
        &runtime.config.data_dir.join("review-upgrade"),
    )
    .unwrap();
    assert_eq!(serde_json::to_value(result).unwrap()["restore"], "ok");
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    assert_eq!(
        scalar(
            &restarted,
            "SELECT rowid FROM review_updates WHERE case_id='old'"
        )
        .await,
        37
    );
    assert_eq!(
        scalar(
            &restarted,
            "SELECT attempts FROM review_updates WHERE case_id='old'"
        )
        .await,
        3
    );
    assert_eq!(
        scalar(&restarted, "SELECT COUNT(*) FROM training_review_locations").await,
        1
    );
    let view = preview(&restarted, &pending.id).await;
    assert_eq!(view["review"]["can_review"], true);
    assert_eq!(view["review"]["message_known"], false);
}

#[tokio::test]
async fn legacy_callback_location_is_available_in_case_details() {
    let runtime = test_runtime().await;
    let case = case(&runtime, false).await;
    runtime
        .decide_training_review_at(&case.id, "reject", HOST_ID, Some((-1, 42)))
        .await
        .unwrap();
    assert_eq!(
        scalar(&runtime, "SELECT COUNT(*) FROM training_review_locations").await,
        0
    );
    let view = preview(&runtime, &case.id).await;
    assert_eq!(view["review"]["message_known"], true);
    assert_eq!(view["review"]["decision"], "reject");
}

#[tokio::test]
async fn cancelled_review_keeps_guards_until_model_and_receipt_are_committed() {
    let runtime = Arc::new(test_runtime().await);
    let case = case(&runtime, false).await;
    let patch = request(&runtime, &case, "approve").await;
    let held = runtime.user_action_guard(200).await;
    let lock = runtime
        .user_action_locks
        .lock()
        .await
        .get(&200)
        .unwrap()
        .upgrade()
        .unwrap();
    drop(held);
    let (started, ready) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let blocked = runtime.clone();
    let blocker = tokio::spawn(async move {
        blocked
            .with_conn(move |_| {
                let _ = started.send(());
                let _ = wait.recv();
                Ok(())
            })
            .await
            .unwrap()
    });
    ready.await.unwrap();
    let saving = runtime.clone();
    let task = tokio::spawn(async move { saving.review_host_case(HOST_ID, patch).await.unwrap() });
    tokio::time::timeout(Duration::from_secs(5), async {
        while runtime.model.try_lock().is_ok() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    let locked = lock.try_lock().is_err();
    release.send(()).unwrap();
    blocker.await.unwrap();
    assert!(locked);
    let _done = lock.lock().await;
    assert_eq!(runtime.model.lock().await.spam_docs, 1);
    assert_eq!(
        scalar(&runtime, "SELECT COUNT(*) FROM host_case_requests").await,
        1
    );
}
