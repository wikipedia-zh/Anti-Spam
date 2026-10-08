use super::*;

async fn due_now(runtime: &Runtime) {
    runtime
        .with_conn(|conn| {
            conn.execute("UPDATE network_deliveries SET next_attempt_at=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
}

async fn queue_state(runtime: &Runtime, case_id: &str, chat_id: i64) -> (String, i64, bool) {
    let case_id = case_id.to_string();
    runtime.with_conn(move |conn| {
        Ok(conn.query_row("SELECT state,attempts,outcome_unknown FROM network_deliveries WHERE case_id=?1 AND chat_id=?2",
            params![case_id,chat_id], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?)
    }).await.unwrap()
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

#[tokio::test]
async fn approved_training_and_delivery_survive_restart_together() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    runtime
        .decide_training_review(&case.id, "approve", HOST_ID)
        .await
        .unwrap();
    // No callback or API call between committing the decision and restarting.
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    assert_eq!(restarted.model.lock().await.spam_docs, 1);
    assert!(restarted
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_some());
    let telegram = TelegramStub::new(vec![]);
    assert_eq!(
        deliver_network_bans(&telegram.bot, &restarted, None)
            .await
            .unwrap(),
        1
    );
    assert_eq!(calls(&telegram, "banchatmember").len(), 1);
    assert_eq!(
        queue_state(&restarted, &case.id, -300).await,
        ("done".into(), 1, false)
    );
    assert!(!restarted
        .decide_training_review(&case.id, "approve", HOST_ID)
        .await
        .unwrap());
    commit_network_ban(&telegram.bot, &restarted, &case).await;
    assert_eq!(calls(&telegram, "banchatmember").len(), 1);
}

#[tokio::test]
async fn delivery_insert_failure_rolls_back_approval_and_model() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_delivery BEFORE INSERT ON network_deliveries BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    assert!(runtime
        .decide_training_review(&case.id, "approve", HOST_ID)
        .await
        .is_err());
    assert_eq!(runtime.model.lock().await.spam_docs, 0);
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_none());
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM training_reviews", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM training_samples", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            conn.execute_batch("DROP TRIGGER fail_delivery;")?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(runtime
        .decide_training_review(&case.id, "approve", HOST_ID)
        .await
        .unwrap());
}

#[tokio::test]
async fn restart_retries_only_failed_groups_and_respects_backoff() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    for chat in [-300, -400] {
        runtime
            .set_group_module(chat, "netban", true)
            .await
            .unwrap();
    }
    let failing = TelegramStub::with_failures(vec![], vec![("banchatmember".into(), -300)]);
    commit_network_ban(&failing.bot, &runtime, &case).await;
    assert_eq!(
        runtime.list_network_ban_targets(&case.id).await.unwrap(),
        vec![-400]
    );
    assert_eq!(
        queue_state(&runtime, &case.id, -300).await,
        ("pending".into(), 1, false)
    );
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    assert_eq!(
        deliver_network_bans(&telegram.bot, &restarted, None)
            .await
            .unwrap(),
        0
    );
    due_now(&restarted).await;
    assert_eq!(
        deliver_network_bans(&telegram.bot, &restarted, None)
            .await
            .unwrap(),
        1
    );
    assert_eq!(calls(&telegram, "banchatmember")[0]["chat_id"], -300);
    assert_eq!(
        queue_state(&restarted, &case.id, -300).await,
        ("done".into(), 2, false)
    );
}

#[tokio::test]
async fn retry_rechecks_current_group_and_whitelist_settings() {
    for change in [
        "module",
        "group_white",
        "global_white",
        "terminated",
        "test_group",
    ] {
        let mut runtime = test_runtime().await;
        let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
        runtime.persist_case(&case).await.unwrap();
        runtime
            .set_group_module(-300, "netban", true)
            .await
            .unwrap();
        runtime.enqueue_network_deliveries(&case.id).await.unwrap();
        match change {
            "module" => runtime
                .set_group_module(-300, "netban", false)
                .await
                .unwrap(),
            "group_white" => {
                runtime.with_conn(|conn| { conn.execute("INSERT INTO group_whitelist(chat_id,user_id,created_at) VALUES (-300,200,'now')",[])?; Ok(()) }).await.unwrap();
            }
            "global_white" => {
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
            "terminated" => {
                runtime.with_conn(|conn| { conn.execute("INSERT INTO banned_groups(chat_id,reason,created_at) VALUES (-300,'test','now')",[])?; Ok(()) }).await.unwrap();
            }
            "test_group" => runtime.config.test_group_id = Some(-300),
            _ => unreachable!(),
        }
        let telegram = TelegramStub::new(vec![]);
        assert_eq!(
            deliver_network_bans(&telegram.bot, &runtime, None)
                .await
                .unwrap(),
            0,
            "{change}"
        );
        assert!(telegram.requests.lock().unwrap().is_empty());
        assert_eq!(queue_state(&runtime, &case.id, -300).await.0, "cancelled");
    }
}

#[tokio::test]
async fn reversal_cancels_unattempted_work_without_unbanning_unrelated_groups() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    runtime.enqueue_network_deliveries(&case.id).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    reverse_ban_case(&telegram.bot, &runtime, case.clone(), HOST_ID, "Host")
        .await
        .unwrap();
    assert_eq!(calls(&telegram, "unbanchatmember").len(), 1);
    assert_eq!(calls(&telegram, "unbanchatmember")[0]["chat_id"], -100);
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    // A stale caller cannot resurrect a reversed case.
    commit_network_ban(&telegram.bot, &restarted, &case).await;
    assert_eq!(
        deliver_network_bans(&telegram.bot, &restarted, None)
            .await
            .unwrap(),
        0
    );
    assert!(calls(&telegram, "banchatmember").is_empty());
}

#[tokio::test]
async fn lost_acknowledgement_remains_reversible_after_a_later_api_failure() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    runtime.enqueue_network_deliveries(&case.id).await.unwrap();
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_ack BEFORE INSERT ON network_ban_targets BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    assert!(deliver_network_bans(&telegram.bot, &runtime, None)
        .await
        .is_err());
    assert_eq!(calls(&telegram, "banchatmember").len(), 1);
    assert_eq!(
        queue_state(&runtime, &case.id, -300).await,
        ("pending".into(), 1, true)
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
    let failing = TelegramStub::with_failures(vec![], vec![("banchatmember".into(), -300)]);
    deliver_network_bans(&failing.bot, &restarted, None)
        .await
        .unwrap();
    assert!(
        queue_state(&restarted, &case.id, -300).await.2,
        "a later rejection cannot disprove the earlier ban"
    );
    reverse_ban_case(&telegram.bot, &restarted, case.clone(), HOST_ID, "Host")
        .await
        .unwrap();
    let unbans = calls(&telegram, "unbanchatmember");
    assert_eq!(unbans.len(), 2);
    assert!(unbans
        .iter()
        .any(|r| r["chat_id"] == -300 && r["only_if_banned"] == true));
    assert_eq!(queue_state(&restarted, &case.id, -300).await.0, "cancelled");
}

#[tokio::test]
async fn concurrent_delivery_and_reversal_never_reban_after_reversal() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    runtime.enqueue_network_deliveries(&case.id).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    let (first, second, reversal) = tokio::join!(
        deliver_network_bans(&telegram.bot, &runtime, None),
        deliver_network_bans(&telegram.bot, &runtime, None),
        reverse_ban_case(&telegram.bot, &runtime, case.clone(), HOST_ID, "Host"),
    );
    first.unwrap();
    second.unwrap();
    reversal.unwrap();
    assert!(calls(&telegram, "banchatmember").len() <= 1);
    assert_eq!(
        deliver_network_bans(&telegram.bot, &runtime, None)
            .await
            .unwrap(),
        0
    );
    let requests = telegram.requests.lock().unwrap();
    if let Some(ban) = requests
        .iter()
        .position(|(method, _)| method == "banchatmember")
    {
        let unban = requests
            .iter()
            .position(|(method, args)| method == "unbanchatmember" && args["chat_id"] == -300)
            .unwrap();
        assert!(ban < unban);
    }
}

#[tokio::test]
async fn rate_limit_survives_restart_and_pauses_reversals_too() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    for chat in [-300, -400] {
        runtime
            .set_group_module(chat, "netban", true)
            .await
            .unwrap();
    }
    runtime.enqueue_network_deliveries(&case.id).await.unwrap();
    let limited = TelegramStub::with_api_errors(
        vec![],
        vec![(
            "banchatmember".into(),
            -400,
            serde_json::json!({"ok":false,"error_code":429,"description":"Too Many Requests: retry after 120","parameters":{"retry_after":120}}),
        )],
    );
    assert_eq!(
        deliver_network_bans(&limited.bot, &runtime, None)
            .await
            .unwrap(),
        1
    );
    assert_eq!(calls(&limited, "banchatmember").len(), 1);
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    due_now(&restarted).await;
    assert_eq!(
        deliver_network_bans(&telegram.bot, &restarted, None)
            .await
            .unwrap(),
        0
    );
    assert!(
        reverse_ban_case(&telegram.bot, &restarted, case.clone(), HOST_ID, "Host")
            .await
            .is_err()
    );
    assert!(telegram.requests.lock().unwrap().is_empty());
    restarted
        .with_conn(|conn| {
            conn.execute("UPDATE telegram_retry_state SET not_before=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
    assert_eq!(
        crate::reversal_retry::retry_due_reversals(&telegram.bot, &restarted)
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        deliver_network_bans(&telegram.bot, &restarted, None)
            .await
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn migration_keeps_existing_reversal_cooldown() {
    let runtime = test_runtime().await;
    runtime.with_conn(|conn| {
        conn.execute_batch("DROP TABLE network_deliveries; ALTER TABLE telegram_retry_state RENAME TO reversal_retry_state;
            UPDATE reversal_retry_state SET not_before=12345678900; PRAGMA user_version=19;")?;
        Runtime::init_db(conn)?;
        Runtime::init_db(conn)?;
        assert_eq!(conn.query_row("SELECT not_before FROM telegram_retry_state",[],|r| r.get::<_,i64>(0))?,12345678900);
        assert_eq!(conn.query_row("PRAGMA user_version",[],|r| r.get::<_,i64>(0))?,38);
        Ok(())
    }).await.unwrap();
}
