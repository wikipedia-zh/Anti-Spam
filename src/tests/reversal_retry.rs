use super::*;
use crate::reversal_retry::retry_due_reversals;

async fn due_now(runtime: &Runtime) {
    runtime
        .with_conn(|conn| {
            conn.execute("UPDATE reversal_retries SET next_attempt_at=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
}

async fn queued(runtime: &Runtime) -> i64 {
    runtime
        .with_conn(|conn| {
            Ok(conn.query_row("SELECT COUNT(*) FROM reversal_retries", [], |r| r.get(0))?)
        })
        .await
        .unwrap()
}

fn unban_requests(telegram: &TelegramStub) -> Vec<serde_json::Value> {
    telegram
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter(|(method, _)| method == "unbanchatmember")
        .map(|(_, args)| args.clone())
        .collect()
}

#[tokio::test]
async fn worker_resumes_failed_targets_after_restart_and_keeps_the_operator() {
    let runtime = test_runtime().await;
    let mut case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    case.actor_user_id = Some(888);
    runtime.persist_case(&case).await.unwrap();
    runtime
        .record_network_ban_target(&case.id, -300)
        .await
        .unwrap();
    let failing = TelegramStub::with_failures(vec![], vec![("unbanchatmember".into(), -300)]);
    assert!(
        reverse_ban_case(&failing.bot, &runtime, case.clone(), 555, "Reviewer")
            .await
            .is_err()
    );
    assert_eq!(
        runtime.list_network_ban_targets(&case.id).await.unwrap(),
        vec![-300]
    );
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let success = TelegramStub::new(vec![]);
    assert_eq!(
        retry_due_reversals(&success.bot, &restarted).await.unwrap(),
        0
    );
    due_now(&restarted).await;
    assert_eq!(
        retry_due_reversals(&success.bot, &restarted).await.unwrap(),
        1
    );
    let finished = restarted.load_case(&case.id).await.unwrap().unwrap();
    assert_eq!(finished.status, "reversed");
    assert_eq!(finished.actor_user_id, Some(555));
    assert_eq!(finished.actor_name.as_deref(), Some("Reviewer"));
    assert_eq!(queued(&restarted).await, 0);
    let requests = unban_requests(&success);
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["chat_id"], -300);
    assert_eq!(requests[0]["only_if_banned"], true);
}

#[tokio::test]
async fn queue_failure_rolls_back_reversal_intent_before_contacting_telegram() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_queue BEFORE INSERT ON reversal_retries BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    assert!(
        reverse_ban_case(&telegram.bot, &runtime, case.clone(), HOST_ID, "Host")
            .await
            .is_err()
    );
    assert_eq!(
        runtime.load_case(&case.id).await.unwrap().unwrap().status,
        case.status
    );
    assert!(runtime
        .list_network_ban_targets(&case.id)
        .await
        .unwrap()
        .is_empty());
    assert_eq!(queued(&runtime).await, 0);
    assert!(telegram.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn rate_limit_pauses_the_queue_across_restart_and_manual_retries() {
    let runtime = test_runtime().await;
    let failing = TelegramStub::with_failures(vec![], vec![("unbanchatmember".into(), -100)]);
    let mut cases = Vec::new();
    for user in [200, 201] {
        let case = dummy_case(ActionKind::SpamBan, -100, user, Utc::now());
        runtime.persist_case(&case).await.unwrap();
        assert!(
            reverse_ban_case(&failing.bot, &runtime, case.clone(), HOST_ID, "Host")
                .await
                .is_err()
        );
        cases.push(case);
    }
    due_now(&runtime).await;
    let limited = TelegramStub::with_api_errors(
        vec![],
        vec![(
            "unbanchatmember".into(),
            -100,
            serde_json::json!({"ok":false,"error_code":429,"description":"Too Many Requests","parameters":{"retry_after":120}}),
        )],
    );
    let before = Utc::now().timestamp();
    assert_eq!(
        retry_due_reversals(&limited.bot, &runtime).await.unwrap(),
        1
    );
    assert_eq!(unban_requests(&limited).len(), 1);
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let until: i64 = restarted
        .with_conn(|conn| {
            Ok(conn.query_row(
                "SELECT not_before FROM telegram_retry_state WHERE id=1",
                [],
                |r| r.get(0),
            )?)
        })
        .await
        .unwrap();
    assert!(until >= before + 120);
    let success = TelegramStub::new(vec![]);
    assert_eq!(
        retry_due_reversals(&success.bot, &restarted).await.unwrap(),
        0
    );
    assert!(
        reverse_ban_case(&success.bot, &restarted, cases[0].clone(), HOST_ID, "Host")
            .await
            .is_err()
    );
    assert!(unban_requests(&success).is_empty());
    restarted
        .with_conn(|conn| {
            conn.execute("UPDATE telegram_retry_state SET not_before=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
    due_now(&restarted).await;
    assert_eq!(
        retry_due_reversals(&success.bot, &restarted).await.unwrap(),
        2
    );
    assert_eq!(queued(&restarted).await, 0);
}

#[tokio::test]
async fn completion_write_failure_does_not_repeat_successful_unbans() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    train_spam(&runtime, "casino gambling", Some(&case.id))
        .await
        .unwrap();
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_finish BEFORE UPDATE ON cases WHEN NEW.status='reversed' BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    assert!(
        reverse_ban_case(&telegram.bot, &runtime, case.clone(), 555, "Reviewer")
            .await
            .is_err()
    );
    assert_eq!(queued(&runtime).await, 1);
    assert!(runtime
        .list_network_ban_targets(&case.id)
        .await
        .unwrap()
        .is_empty());
    runtime
        .with_conn(|conn| {
            conn.execute_batch("DROP TRIGGER fail_finish;")?;
            Ok(())
        })
        .await
        .unwrap();
    due_now(&runtime).await;
    assert_eq!(
        retry_due_reversals(&telegram.bot, &runtime).await.unwrap(),
        1
    );
    assert_eq!(unban_requests(&telegram).len(), 1);
    assert_eq!(queued(&runtime).await, 0);
    assert_eq!(runtime.model.lock().await.spam_docs, 0);
    assert_eq!(
        runtime
            .load_case(&case.id)
            .await
            .unwrap()
            .unwrap()
            .actor_user_id,
        Some(555)
    );
}

#[tokio::test]
async fn uncertain_unban_is_retried_with_only_if_banned() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_ack BEFORE DELETE ON network_ban_targets BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    assert!(
        reverse_ban_case(&telegram.bot, &runtime, case.clone(), HOST_ID, "Host")
            .await
            .is_err()
    );
    runtime
        .with_conn(|conn| {
            conn.execute_batch("DROP TRIGGER fail_ack;")?;
            Ok(())
        })
        .await
        .unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due_now(&restarted).await;
    assert_eq!(
        retry_due_reversals(&telegram.bot, &restarted)
            .await
            .unwrap(),
        1
    );
    let requests = unban_requests(&telegram);
    assert_eq!(requests.len(), 2);
    assert!(requests.iter().all(|args| args["only_if_banned"] == true));
    assert_eq!(queued(&restarted).await, 0);
}

#[tokio::test]
async fn retry_keeps_an_independent_ban_added_while_it_was_waiting() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    let failing = TelegramStub::with_failures(vec![], vec![("unbanchatmember".into(), -100)]);
    assert!(
        reverse_ban_case(&failing.bot, &runtime, case.clone(), HOST_ID, "Host")
            .await
            .is_err()
    );
    let other = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&other).await.unwrap();
    due_now(&runtime).await;
    let telegram = TelegramStub::new(vec![]);
    assert_eq!(
        retry_due_reversals(&telegram.bot, &runtime).await.unwrap(),
        1
    );
    assert!(unban_requests(&telegram).is_empty());
    assert_eq!(
        runtime
            .find_active_ban_in_chat(-100, 200)
            .await
            .unwrap()
            .unwrap()
            .id,
        other.id
    );
    assert_eq!(queued(&runtime).await, 0);
}

#[tokio::test]
async fn manual_and_worker_retries_do_not_overlap() {
    let runtime = Arc::new(test_runtime().await);
    let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    let failing = TelegramStub::with_failures(vec![], vec![("unbanchatmember".into(), -100)]);
    assert!(
        reverse_ban_case(&failing.bot, &runtime, case.clone(), 555, "Reviewer")
            .await
            .is_err()
    );
    due_now(&runtime).await;
    let telegram = TelegramStub::new(vec![]);
    let (worker, manual) = tokio::join!(
        retry_due_reversals(&telegram.bot, &runtime),
        reverse_ban_case(&telegram.bot, &runtime, case.clone(), HOST_ID, "Host")
    );
    worker.unwrap();
    manual.unwrap();
    assert_eq!(unban_requests(&telegram).len(), 1);
    assert_eq!(
        runtime
            .load_case(&case.id)
            .await
            .unwrap()
            .unwrap()
            .actor_user_id,
        Some(555)
    );
    assert_eq!(queued(&runtime).await, 0);
}

#[tokio::test]
async fn startup_worker_recovers_legacy_pending_reversals_without_inventing_an_actor() {
    let runtime = test_runtime().await;
    let mut case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    case.status = "reversal_pending".into();
    case.actor_user_id = Some(888);
    case.actor_name = Some("Original banner".into());
    runtime.persist_case(&case).await.unwrap();
    runtime
        .record_network_ban_target(&case.id, -300)
        .await
        .unwrap();
    runtime.with_conn(|conn| {
        conn.execute_batch("DROP TABLE reversal_retries; DROP TABLE telegram_retry_state; PRAGMA user_version=18;")?;
        Ok(())
    }).await.unwrap();
    let restarted = Arc::new(Runtime::load(runtime.config.clone()).await.unwrap());
    let telegram = TelegramStub::new(vec![]);
    let worker = spawn_reversal_worker(telegram.bot.clone(), restarted.clone());
    let result = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if restarted.load_case(&case.id).await.unwrap().unwrap().status == "reversed" {
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
        let _guard = restarted.review_guard(&case.id).await;
    })
    .await;
    worker.abort();
    let _ = worker.await;
    result.unwrap();
    let finished = restarted.load_case(&case.id).await.unwrap().unwrap();
    assert_eq!(finished.actor_user_id, None);
    assert_eq!(finished.actor_name, None);
    assert_eq!(unban_requests(&telegram).len(), 1);
    assert_eq!(unban_requests(&telegram)[0]["chat_id"], -300);
    assert_eq!(queued(&restarted).await, 0);
}

#[tokio::test]
async fn stale_queue_entries_do_not_change_closed_cases() {
    let runtime = test_runtime().await;
    let mut case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    let failing = TelegramStub::with_failures(vec![], vec![("unbanchatmember".into(), -100)]);
    assert!(
        reverse_ban_case(&failing.bot, &runtime, case.clone(), HOST_ID, "Host")
            .await
            .is_err()
    );
    case.action = ActionKind::Unbanned;
    case.status = "reversed".into();
    runtime.persist_case(&case).await.unwrap();
    due_now(&runtime).await;
    let telegram = TelegramStub::new(vec![]);
    assert_eq!(
        retry_due_reversals(&telegram.bot, &runtime).await.unwrap(),
        0
    );
    assert!(telegram.requests.lock().unwrap().is_empty());
    assert_eq!(queued(&runtime).await, 0);
}

#[tokio::test]
async fn repeated_failures_back_off_and_keep_the_job() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    let failing = TelegramStub::with_failures(vec![], vec![("unbanchatmember".into(), -100)]);
    for seconds in [60, 120] {
        let before = Utc::now().timestamp();
        assert!(
            reverse_ban_case(&failing.bot, &runtime, case.clone(), HOST_ID, "Host")
                .await
                .is_err()
        );
        let (next, error): (i64, String) = runtime
            .with_conn(|conn| {
                Ok(conn.query_row(
                    "SELECT next_attempt_at,last_error FROM reversal_retries",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )?)
            })
            .await
            .unwrap();
        assert!(next >= before + seconds);
        assert!(next <= Utc::now().timestamp() + seconds);
        assert!(error.contains("injected failure"));
    }
    runtime
        .with_conn(|conn| {
            conn.execute("UPDATE reversal_retries SET attempts=30", [])?;
            Ok(())
        })
        .await
        .unwrap();
    let before = Utc::now().timestamp();
    assert!(
        reverse_ban_case(&failing.bot, &runtime, case.clone(), HOST_ID, "Host")
            .await
            .is_err()
    );
    let next: i64 = runtime
        .with_conn(|conn| {
            Ok(
                conn.query_row("SELECT next_attempt_at FROM reversal_retries", [], |r| {
                    r.get(0)
                })?,
            )
        })
        .await
        .unwrap();
    assert!(next >= before + 3600);
    assert!(next <= Utc::now().timestamp() + 3600);
    assert_eq!(queued(&runtime).await, 1);
}
