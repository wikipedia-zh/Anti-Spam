use super::*;
use crate::origin_retry::retry_origin_bans;

async fn due_now(runtime: &Runtime) {
    runtime.with_conn(|conn| {
        conn.execute_batch("UPDATE origin_ban_jobs SET next_attempt_at=0; UPDATE telegram_retry_state SET not_before=0;")?;
        Ok(())
    }).await.unwrap();
}

fn calls(telegram: &TelegramStub, method: &str) -> Vec<serde_json::Value> {
    telegram
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter(|(m, _)| m == method)
        .map(|(_, args)| args.clone())
        .collect()
}

async fn state(runtime: &Runtime, id: &str) -> (String, bool, bool) {
    let id = id.to_string();
    runtime
        .with_conn(move |conn| {
            Ok(conn.query_row(
                "SELECT state,ban_done,outcome_unknown FROM origin_ban_jobs WHERE case_id=?1",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )?)
        })
        .await
        .unwrap()
}

#[tokio::test]
async fn failed_original_ban_resumes_after_restart_without_repeating_deletion() {
    let runtime = test_runtime().await;
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    let mut case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    case.source_message_id = Some(1);
    case.model_score = Some(0.99);
    let failing = TelegramStub::with_failures(vec![], vec![("banchatmember".into(), -100)]);
    assert!(
        !execute_auto_ban(&failing.bot, &runtime, case.clone(), "test")
            .await
            .unwrap()
    );
    assert_eq!(
        state(&runtime, &case.id).await,
        ("pending".into(), false, false)
    );
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_none());
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due_now(&restarted).await;
    let success = TelegramStub::new(vec![]);
    retry_origin_bans(&success.bot, &restarted).await.unwrap();
    assert_eq!(
        state(&restarted, &case.id).await,
        ("done".into(), true, false)
    );
    assert!(calls(&success, "deletemessage").is_empty());
    assert_eq!(calls(&success, "banchatmember").len(), 1);
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_some());
    assert_eq!(
        calls(&success, "editmessagetext").len(),
        1,
        "reuse the failed case's log"
    );
    deliver_network_bans(&success.bot, &restarted, None)
        .await
        .unwrap();
    assert!(calls(&success, "banchatmember")
        .iter()
        .any(|args| args["chat_id"] == -300));
    due_now(&restarted).await;
    retry_origin_bans(&success.bot, &restarted).await.unwrap();
    assert_eq!(calls(&success, "banchatmember").len(), 2);
}

#[tokio::test]
async fn failed_log_retries_without_rebanning_or_losing_network_work() {
    let runtime = test_runtime().await;
    let mut case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    case.model_score = Some(0.99);
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    let failing = TelegramStub::with_failures(vec![], vec![("sendmessage".into(), -1)]);
    assert!(
        execute_auto_ban(&failing.bot, &runtime, case.clone(), "test")
            .await
            .unwrap()
    );
    assert!(calls(&failing, "banchatmember")
        .iter()
        .any(|args| args["chat_id"] == -300));
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due_now(&restarted).await;
    let success = TelegramStub::new(vec![]);
    retry_origin_bans(&success.bot, &restarted).await.unwrap();
    assert!(calls(&success, "banchatmember").is_empty());
    assert_eq!(state(&restarted, &case.id).await.0, "done");
    assert!(restarted
        .load_case(&case.id)
        .await
        .unwrap()
        .unwrap()
        .log_message_id
        .is_some());
}

#[tokio::test]
async fn failed_database_acknowledgement_preserves_uncertainty_and_reversal_cancels_retry() {
    let runtime = test_runtime().await;
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_ban_ack BEFORE UPDATE OF status ON cases WHEN NEW.status='auto_banned'
            BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    let telegram = TelegramStub::new(vec![]);
    assert!(
        execute_auto_ban(&telegram.bot, &runtime, case.clone(), "test")
            .await
            .is_err()
    );
    assert_eq!(calls(&telegram, "banchatmember").len(), 1);
    assert_eq!(
        state(&runtime, &case.id).await,
        ("pending".into(), false, true)
    );
    assert!(calls(&telegram, "sendmessage").is_empty());
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    reverse_ban_case(&telegram.bot, &restarted, case.clone(), HOST_ID, "Host")
        .await
        .unwrap();
    assert_eq!(calls(&telegram, "unbanchatmember").len(), 1);
    due_now(&restarted).await;
    retry_origin_bans(&telegram.bot, &restarted).await.unwrap();
    execute_auto_ban(&telegram.bot, &restarted, case.clone(), "replay")
        .await
        .unwrap();
    assert_eq!(calls(&telegram, "banchatmember").len(), 1);
    assert_eq!(
        restarted.load_case(&case.id).await.unwrap().unwrap().status,
        "reversed"
    );
}

#[tokio::test]
async fn queue_and_case_are_atomic_and_historical_failures_are_not_adopted() {
    let runtime = test_runtime().await;
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_origin BEFORE INSERT ON origin_ban_jobs BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    let telegram = TelegramStub::new(vec![]);
    assert!(
        execute_auto_ban(&telegram.bot, &runtime, case.clone(), "test")
            .await
            .is_err()
    );
    assert!(runtime.load_case(&case.id).await.unwrap().is_none());
    assert!(telegram.requests.lock().unwrap().is_empty());
    let mut historical = case;
    historical.status = "ban_failed".into();
    runtime.persist_case(&historical).await.unwrap();
    execute_auto_ban(&telegram.bot, &runtime, historical.clone(), "test")
        .await
        .unwrap();
    assert!(telegram.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn retry_checks_new_exemptions_and_fails_closed_on_member_lookup_error() {
    for exemption in ["admin", "whitelist", "lookup_error"] {
        let runtime = test_runtime().await;
        runtime.delay_telegram_queue(300).await.unwrap();
        let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
        let idle = TelegramStub::new(vec![]);
        execute_auto_ban(&idle.bot, &runtime, case.clone(), "test")
            .await
            .unwrap();
        assert!(idle.requests.lock().unwrap().is_empty());
        if exemption == "whitelist" {
            runtime
                .with_conn(|conn| {
                    conn.execute(
                        "INSERT INTO global_whitelist(user_id,created_at) VALUES (200,'now')",
                        [],
                    )?;
                    Ok(())
                })
                .await
                .unwrap();
        }
        let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
        due_now(&restarted).await;
        let telegram = TelegramStub::with_failures(
            if exemption == "admin" {
                vec![200]
            } else {
                vec![]
            },
            if exemption == "lookup_error" {
                vec![("getchatmember".into(), -100)]
            } else {
                vec![]
            },
        );
        retry_origin_bans(&telegram.bot, &restarted).await.unwrap();
        assert!(calls(&telegram, "banchatmember").is_empty());
        assert!(calls(&telegram, "deletemessage").is_empty());
        assert_eq!(
            state(&restarted, &case.id).await.0,
            if exemption == "lookup_error" {
                "pending"
            } else {
                "cancelled"
            }
        );
    }
}

#[tokio::test]
async fn concurrent_replays_do_not_repeat_a_successful_ban() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    let telegram = TelegramStub::new(vec![]);
    let (a, b) = tokio::join!(
        execute_auto_ban(&telegram.bot, &runtime, case.clone(), "test"),
        execute_auto_ban(&telegram.bot, &runtime, case.clone(), "test")
    );
    a.unwrap();
    b.unwrap();
    assert_eq!(calls(&telegram, "banchatmember").len(), 1);
    assert_eq!(state(&runtime, &case.id).await.0, "done");
}

#[tokio::test]
async fn deletion_retry_keeps_the_successful_ban_and_finishes_its_notice() {
    let runtime = test_runtime().await;
    let mut case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    case.source_message_id = Some(1);
    let failed = TelegramStub::with_failures(vec![], vec![("deletemessage".into(), -100)]);
    assert!(
        execute_auto_ban(&failed.bot, &runtime, case.clone(), "test")
            .await
            .unwrap()
    );
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due_now(&restarted).await;
    let success = TelegramStub::new(vec![]);
    retry_origin_bans(&success.bot, &restarted).await.unwrap();
    assert!(calls(&success, "banchatmember").is_empty());
    assert_eq!(calls(&success, "deletemessage").len(), 1);
    assert_eq!(
        restarted.load_case(&case.id).await.unwrap().unwrap().status,
        "auto_banned"
    );
    assert_eq!(state(&restarted, &case.id).await.0, "done");
}

#[tokio::test]
async fn rate_limit_persists_and_stops_all_further_api_calls() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    let limited = TelegramStub::with_api_errors(
        vec![],
        vec![(
            "banchatmember".into(),
            -100,
            serde_json::json!({"ok":false,"error_code":429,"description":"Too Many Requests: retry after 120","parameters":{"retry_after":120}}),
        )],
    );
    assert!(
        !execute_auto_ban(&limited.bot, &runtime, case.clone(), "test")
            .await
            .unwrap()
    );
    assert!(calls(&limited, "sendmessage").is_empty());
    assert_eq!(
        state(&runtime, &case.id).await,
        ("pending".into(), false, false)
    );
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    restarted
        .with_conn(|conn| {
            conn.execute("UPDATE origin_ban_jobs SET next_attempt_at=0", [])?;
            assert!(
                conn.query_row(
                    "SELECT not_before FROM telegram_retry_state WHERE id=1",
                    [],
                    |r| r.get::<_, i64>(0)
                )? > Utc::now().timestamp()
            );
            Ok(())
        })
        .await
        .unwrap();
    let success = TelegramStub::new(vec![]);
    assert_eq!(
        retry_origin_bans(&success.bot, &restarted).await.unwrap(),
        0
    );
    assert!(success.requests.lock().unwrap().is_empty());
    due_now(&restarted).await;
    retry_origin_bans(&success.bot, &restarted).await.unwrap();
    assert_eq!(state(&restarted, &case.id).await.0, "done");
}

#[tokio::test]
async fn retry_recovers_a_successful_ban_with_a_lost_database_acknowledgement() {
    let runtime = test_runtime().await;
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_ban_ack BEFORE UPDATE OF status ON cases WHEN NEW.status='auto_banned'
            BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    let telegram = TelegramStub::new(vec![]);
    assert!(
        execute_auto_ban(&telegram.bot, &runtime, case.clone(), "test")
            .await
            .is_err()
    );
    assert!(
        runtime
            .has_other_ban_in_chat("another-case", -100, 200)
            .await
            .unwrap(),
        "an uncertain independent ban must survive another case's reversal"
    );
    runtime
        .with_conn(|conn| {
            conn.execute_batch("DROP TRIGGER fail_ban_ack;")?;
            Ok(())
        })
        .await
        .unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due_now(&restarted).await;
    retry_origin_bans(&telegram.bot, &restarted).await.unwrap();
    assert_eq!(
        state(&restarted, &case.id).await,
        ("done".into(), true, false)
    );
    assert_eq!(
        restarted.load_case(&case.id).await.unwrap().unwrap().status,
        "auto_banned"
    );
}

#[tokio::test]
async fn changed_module_or_threshold_cancels_unfinished_enforcement() {
    for reason in ["VOICE", "ML", "REGEX@9999"] {
        let runtime = test_runtime().await;
        runtime.delay_telegram_queue(300).await.unwrap();
        let mut case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
        case.matched_rule_pattern = Some(reason.into());
        case.model_score = if reason == "ML" { Some(0.6) } else { None };
        let telegram = TelegramStub::new(vec![]);
        execute_auto_ban(&telegram.bot, &runtime, case.clone(), "test")
            .await
            .unwrap();
        due_now(&runtime).await;
        retry_origin_bans(&telegram.bot, &runtime).await.unwrap();
        assert_eq!(state(&runtime, &case.id).await.0, "cancelled");
        assert!(calls(&telegram, "banchatmember").is_empty());
    }
}

#[tokio::test]
async fn notification_and_exchange_retries_preserve_completed_deliveries() {
    for failed_chat in [-100, -500] {
        let runtime = test_runtime().await;
        runtime.set_exchange_channel(-500).await;
        let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
        let failed = TelegramStub::with_failures(vec![], vec![("sendmessage".into(), failed_chat)]);
        assert!(
            execute_auto_ban(&failed.bot, &runtime, case.clone(), "test")
                .await
                .unwrap()
        );
        assert_eq!(state(&runtime, &case.id).await.0, "pending");
        let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
        due_now(&restarted).await;
        let success = TelegramStub::new(vec![]);
        retry_origin_bans(&success.bot, &restarted).await.unwrap();
        assert!(calls(&success, "banchatmember").is_empty());
        let sent = calls(&success, "sendmessage");
        assert!(
            !sent.iter().any(|args| args["chat_id"] == -1),
            "known log must not be posted again"
        );
        assert_eq!(
            sent.iter().filter(|args| args["chat_id"] == -100).count(),
            usize::from(failed_chat == -100)
        );
        assert_eq!(
            sent.iter().filter(|args| args["chat_id"] == -500).count(),
            1
        );
        assert_eq!(state(&restarted, &case.id).await.0, "done");
    }
}
