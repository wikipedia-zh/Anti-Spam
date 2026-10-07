use super::*;
use crate::host_cases::{Outcome, Query, Reverse};

async fn preview(runtime: &Runtime, id: &str) -> serde_json::Value {
    runtime
        .host_case(Query {
            case_id: id.into(),
            offset: 0,
        })
        .await
        .unwrap()
        .unwrap()
}
async fn request(runtime: &Runtime, id: &str) -> Reverse {
    Reverse {
        request_id: Uuid::new_v4().to_string(),
        case_id: id.into(),
        target_user_id: preview(runtime, id).await["case"]["target_user_id"]
            .as_i64()
            .unwrap(),
        expected_revision: preview(runtime, id).await["revision"]
            .as_str()
            .unwrap()
            .into(),
    }
}

#[tokio::test]
async fn preview_counts_uncertain_pending_and_independent_bans_and_redacts_evidence() {
    let runtime = test_runtime().await;
    let mut case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    case.evidence_text = runtime.config.bot_token.clone();
    runtime.persist_case(&case).await.unwrap();
    let other = dummy_case(ActionKind::SpamBan, -300, 200, Utc::now());
    runtime.persist_case(&other).await.unwrap();
    runtime
        .record_network_ban_target(&case.id, -300)
        .await
        .unwrap();
    let id = case.id.clone();
    runtime.with_conn(move |conn| {
        conn.execute("INSERT INTO network_deliveries(case_id,chat_id,outcome_unknown) VALUES (?1,-400,1),(?1,-500,0)",[id])?;
        Ok(())
    }).await.unwrap();
    let data = preview(&runtime, &case.id).await;
    assert_eq!(data["unban_count"], 2);
    assert_eq!(data["retained_count"], 1);
    assert_eq!(data["cancel_count"], 2);
    assert!(!data.to_string().contains(&runtime.config.bot_token));
    assert_eq!(data["targets"][0]["reversal_action"], "cancel");
    assert_eq!(data["targets"][1]["reversal_action"], "unban");
    assert_eq!(data["targets"][2]["reversal_action"], "retain");
    assert!(matches!(
        runtime
            .reverse_host_case(HOST_ID, request(&runtime, &case.id).await)
            .await
            .unwrap(),
        Outcome::Saved(_)
    ));
    let telegram = TelegramStub::new(vec![]);
    crate::reversal_retry::retry_due_reversals(&telegram.bot, &runtime)
        .await
        .unwrap();
    let calls = telegram.requests.lock().unwrap();
    let unbans: Vec<_> = calls
        .iter()
        .filter(|(method, _)| method == "unbanchatmember")
        .map(|(_, args)| args["chat_id"].as_i64().unwrap())
        .collect();
    assert!(unbans.contains(&-100));
    assert!(unbans.contains(&-400));
    assert!(!unbans.contains(&-300));
    assert!(!unbans.contains(&-500));
}

#[tokio::test]
async fn reversal_receipt_survives_restart_and_worker_retries_only_failed_targets() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    train_spam(&runtime, "test casino", Some(&case.id))
        .await
        .unwrap();
    runtime
        .record_network_ban_target(&case.id, -300)
        .await
        .unwrap();
    let patch = request(&runtime, &case.id).await;
    let (a, b) = tokio::join!(
        runtime.reverse_host_case(HOST_ID, patch.clone()),
        runtime.reverse_host_case(HOST_ID, patch.clone())
    );
    assert!(matches!(a.unwrap(), Outcome::Saved(_)));
    assert!(matches!(b.unwrap(), Outcome::Saved(_)));
    runtime
        .with_conn(|conn| {
            for table in [
                "host_case_requests",
                "reversal_retries",
                "maintainer_actions",
            ] {
                assert_eq!(
                    conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r
                        .get::<_, i64>(0))?,
                    1
                );
            }
            Ok(())
        })
        .await
        .unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let failing = TelegramStub::with_failures(vec![], vec![("unbanchatmember".into(), -300)]);
    assert_eq!(
        crate::reversal_retry::retry_due_reversals(&failing.bot, &restarted)
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        restarted.load_case(&case.id).await.unwrap().unwrap().status,
        "reversal_pending"
    );
    assert_eq!(
        restarted.list_network_ban_targets(&case.id).await.unwrap(),
        vec![-300]
    );
    restarted
        .with_conn(|conn| {
            conn.execute("UPDATE reversal_retries SET next_attempt_at=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
    let success = TelegramStub::new(vec![]);
    assert_eq!(
        crate::reversal_retry::retry_due_reversals(&success.bot, &restarted)
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        restarted.load_case(&case.id).await.unwrap().unwrap().status,
        "reversed"
    );
    assert!(matches!(
        restarted
            .reverse_host_case(HOST_ID, patch.clone())
            .await
            .unwrap(),
        Outcome::Saved(_)
    ));
    let data = preview(&restarted, &case.id).await;
    assert_eq!(data["training_samples"], 0);
    assert_eq!(data["can_reverse"], false);
    assert_eq!(data["unban_count"], 0);
    restarted
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM reversal_retries", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM maintainer_actions", [], |r| r
                    .get::<_, i64>(0))?,
                1
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn changed_impact_requires_new_confirmation_and_failed_receipt_rolls_back() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    let stale = request(&runtime, &case.id).await;
    runtime
        .record_network_ban_target(&case.id, -300)
        .await
        .unwrap();
    assert!(matches!(
        runtime.reverse_host_case(HOST_ID, stale).await.unwrap(),
        Outcome::Conflict
    ));
    let patch = request(&runtime, &case.id).await;
    assert!(matches!(
        runtime.reverse_host_case(300, patch.clone()).await.unwrap(),
        Outcome::Forbidden
    ));
    runtime.with_conn(|conn|{conn.execute_batch("CREATE TRIGGER fail_case_receipt BEFORE INSERT ON host_case_requests BEGIN SELECT RAISE(ABORT,'receipt failed'); END;")?;Ok(())}).await.unwrap();
    assert!(runtime
        .reverse_host_case(HOST_ID, patch.clone())
        .await
        .is_err());
    assert_eq!(
        runtime.load_case(&case.id).await.unwrap().unwrap().status,
        case.status
    );
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM reversal_retries", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM maintainer_actions", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            conn.execute_batch("DROP TRIGGER fail_case_receipt;")?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(matches!(
        runtime
            .reverse_host_case(HOST_ID, patch.clone())
            .await
            .unwrap(),
        Outcome::Saved(_)
    ));
    let mut reused = patch;
    reused.expected_revision = "0".repeat(64);
    assert!(matches!(
        runtime.reverse_host_case(HOST_ID, reused).await.unwrap(),
        Outcome::Conflict
    ));
}

#[tokio::test]
async fn case_details_paginate_without_changing_confirmation_and_reject_non_bans() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::Mute, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    for chat in -140..-100 {
        runtime
            .record_network_ban_target(&case.id, chat)
            .await
            .unwrap();
    }
    let first = preview(&runtime, &case.id).await;
    let second = runtime
        .host_case(Query {
            case_id: case.id.clone(),
            offset: 25,
        })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(first["targets"].as_array().unwrap().len(), 25);
    assert_eq!(second["targets"].as_array().unwrap().len(), 16);
    assert_eq!(first["revision"], second["revision"]);
    assert_eq!(first["can_reverse"], false);
    assert!(matches!(
        runtime
            .reverse_host_case(HOST_ID, request(&runtime, &case.id).await)
            .await
            .unwrap(),
        Outcome::Invalid
    ));
}

#[tokio::test]
async fn cancelled_reversal_request_keeps_locks_through_commit() {
    let runtime = Arc::new(test_runtime().await);
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    let patch = request(&runtime, &case.id).await;
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
            .unwrap();
    });
    ready.await.unwrap();
    let saving = runtime.clone();
    let task = tokio::spawn(async move { saving.reverse_host_case(HOST_ID, patch).await.unwrap() });
    tokio::time::timeout(Duration::from_secs(5), async {
        while lock.try_lock().is_ok() {
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
    assert_eq!(
        runtime.load_case(&case.id).await.unwrap().unwrap().status,
        "reversal_pending"
    );
}
