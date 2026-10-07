use super::*;

fn review_callback(case_id: &str, kind: &str, decision: &str) -> CallbackQuery {
    serde_json::from_value(serde_json::json!({
        "id": Uuid::new_v4().to_string(),
        "from": {"id": HOST_ID, "is_bot": false, "first_name": "Host"},
        "chat_instance": "review-channel", "data": format!("{kind}:{decision}:{case_id}"),
        "message": {
            "message_id": 10, "date": 1, "text": "Review",
            "chat": {"id": -1, "type": "channel", "title": "Reviews"}
        }
    }))
    .unwrap()
}

#[tokio::test]
async fn reviewing_one_case_does_not_block_an_unrelated_case() {
    let runtime = Arc::new(test_runtime().await);
    let blocked = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    let ready = dummy_case(ActionKind::SpamBan, -300, 201, Utc::now());
    runtime.persist_case(&blocked).await.unwrap();
    runtime.persist_case(&ready).await.unwrap();
    let _guard = runtime.review_guard(&blocked.id).await;
    let telegram = TelegramStub::new(vec![]);
    tokio::time::timeout(
        Duration::from_secs(2),
        handle_callback(
            telegram.bot.clone(),
            runtime.clone(),
            review_callback(&ready.id, "train", "approve"),
        ),
    )
    .await
    .expect("an unrelated case must remain available")
    .unwrap();
    assert!(runtime
        .find_active_network_ban(201)
        .await
        .unwrap()
        .is_some());
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_none());
    assert_eq!(runtime.model.lock().await.spam_docs, 1);
}

#[tokio::test]
async fn concurrent_training_callbacks_apply_one_decision_despite_failed_keyboard_edits() {
    let runtime = Arc::new(test_runtime().await);
    let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    let telegram = TelegramStub::with_failures(
        vec![],
        vec![
            ("editmessagetext".into(), -1),
            ("editmessagereplymarkup".into(), -1),
        ],
    );
    let (first, second) = tokio::join!(
        handle_callback(
            telegram.bot.clone(),
            runtime.clone(),
            review_callback(&case.id, "train", "approve")
        ),
        handle_callback(
            telegram.bot.clone(),
            runtime.clone(),
            review_callback(&case.id, "train", "approve")
        )
    );
    first.unwrap();
    second.unwrap();
    let restarted = Arc::new(Runtime::load(runtime.config.clone()).await.unwrap());
    handle_callback(
        telegram.bot.clone(),
        restarted.clone(),
        review_callback(&case.id, "train", "reject"),
    )
    .await
    .unwrap();
    assert_eq!(restarted.model.lock().await.spam_docs, 1);
    assert!(restarted
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_some());
    let edits = telegram
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter(|(method, _)| method == "editmessagetext")
        .count();
    assert_eq!(
        edits, 1,
        "only the winning decision attempts to edit the review"
    );
}

#[tokio::test]
async fn repeated_report_rejection_does_not_repeat_reporter_strikes() {
    let runtime = Arc::new(test_runtime().await);
    let mut case = dummy_case(ActionKind::PendingReport, -100, 200, Utc::now());
    case.status = "pending_review".into();
    case.actor_user_id = Some(444);
    runtime.persist_case(&case).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    handle_callback(
        telegram.bot.clone(),
        runtime.clone(),
        review_callback(&case.id, "review", "reject"),
    )
    .await
    .unwrap();
    let restarted = Arc::new(Runtime::load(runtime.config.clone()).await.unwrap());
    handle_callback(
        telegram.bot.clone(),
        restarted.clone(),
        review_callback(&case.id, "review", "reject"),
    )
    .await
    .unwrap();
    assert_eq!(restarted.report_strikes(444).await, 1);
    assert_eq!(restarted.model.lock().await.ham_docs, 1);
    assert_eq!(
        restarted.load_case(&case.id).await.unwrap().unwrap().action,
        ActionKind::ReportRejected
    );
}

#[test]
fn extreme_scores_are_finite_and_invalid_scores_cannot_ban() {
    assert_eq!(stable_probability(1000.0), 1.0);
    assert_eq!(stable_probability(-1000.0), 0.0);
    assert_eq!(stable_probability(0.0), 0.5);
    for score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 2.0] {
        assert!(!passes_threshold(score, 0.85));
        assert!(!netban_eligible(
            &ActionKind::AutoBan,
            Some(score),
            0.85,
            None
        ));
    }
    assert!(!passes_threshold(1.0, f64::NAN));
    let model = ModelState {
        spam_docs: 100,
        ham_docs: 100,
        spam_tokens: HashMap::from([("casino".into(), 1000)]),
        ham_tokens: HashMap::from([("article".into(), 1000)]),
    };
    let text = "casino ".repeat(500);
    assert!(score_spam_from_text(&model, &text).is_finite());
    assert_eq!(
        score_spam_from_text(&model, &text),
        score_debug_from_text(&model, &text).score
    );
}

#[tokio::test]
async fn failed_training_rolls_back_sample_tokens_counts_and_memory() {
    let runtime = test_runtime().await;
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_words BEFORE INSERT ON word_frequencies BEGIN SELECT RAISE(ABORT, 'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    assert!(train_spam(&runtime, "casino gambling", Some("atomic"))
        .await
        .is_err());
    assert_eq!(runtime.model.lock().await.spam_docs, 0);
    let persisted = runtime.rebuild_model().await.unwrap();
    assert_eq!(persisted.spam_docs, 0);
    assert!(persisted.spam_tokens.is_empty());
    runtime
        .with_conn(|conn| {
            conn.execute_batch("DROP TRIGGER fail_words;")?;
            Ok(())
        })
        .await
        .unwrap();
    train_spam(&runtime, "casino gambling", Some("atomic"))
        .await
        .unwrap();
    assert_eq!(runtime.rebuild_model().await.unwrap().spam_docs, 1);
}

#[tokio::test]
async fn training_retries_are_idempotent_and_conflicting_labels_are_rejected() {
    let runtime = Arc::new(test_runtime().await);
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let runtime = runtime.clone();
        tasks.push(tokio::spawn(async move {
            train_spam(&runtime, "casino gambling", Some("retry")).await
        }));
    }
    for task in tasks {
        task.await.unwrap().unwrap();
    }
    assert_eq!(runtime.model.lock().await.spam_docs, 1);
    assert!(train_ham(&runtime, "casino gambling", Some("retry"))
        .await
        .is_err());
    train_spam(&runtime, "🎉", Some("empty")).await.unwrap();
    let reloaded = Runtime::load(runtime.config.clone()).await.unwrap();
    assert_eq!(reloaded.model.lock().await.spam_docs, 1);
    assert_eq!(reloaded.model.lock().await.ham_docs, 0);
}

#[tokio::test]
async fn training_review_first_decision_survives_restart_and_reversal() {
    for decision in ["approve", "reject"] {
        let runtime = test_runtime().await;
        let mut case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
        case.evidence_text = "casino gambling".to_string();
        runtime.persist_case(&case).await.unwrap();
        assert!(runtime
            .decide_training_review(&case.id, decision, HOST_ID)
            .await
            .unwrap());
        let reloaded = Runtime::load(runtime.config.clone()).await.unwrap();
        assert!(!reloaded
            .decide_training_review(&case.id, "approve", HOST_ID)
            .await
            .unwrap());
        assert!(!reloaded
            .decide_training_review(&case.id, "reject", HOST_ID)
            .await
            .unwrap());
        assert_eq!(
            reloaded.model.lock().await.spam_docs,
            u64::from(decision == "approve")
        );
        case.action = ActionKind::Unbanned;
        case.status = "reversed".to_string();
        reloaded.persist_case(&case).await.unwrap();
        assert!(reloaded
            .decide_training_review(&case.id, "approve", HOST_ID)
            .await
            .is_err());
    }
}

#[tokio::test]
async fn failed_review_does_not_consume_the_decision() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_samples BEFORE INSERT ON training_samples BEGIN SELECT RAISE(ABORT, 'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    assert!(runtime
        .decide_training_review(&case.id, "approve", HOST_ID)
        .await
        .is_err());
    runtime
        .with_conn(|conn| {
            conn.execute_batch("DROP TRIGGER fail_samples;")?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(runtime
        .decide_training_review(&case.id, "approve", HOST_ID)
        .await
        .unwrap());
    assert_eq!(runtime.model.lock().await.spam_docs, 1);
}

#[tokio::test]
async fn failed_auto_ban_does_not_enter_active_bans_or_propagate() {
    let runtime = test_runtime().await;
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    let telegram = TelegramStub::with_failures(vec![], vec![("banchatmember".to_string(), -100)]);
    let mut case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    case.model_score = Some(0.99);
    case.source_message_id = Some(1);
    assert!(
        !execute_auto_ban(&telegram.bot, &runtime, case.clone(), "<b>自動封禁</b>")
            .await
            .unwrap()
    );
    assert_eq!(
        runtime.load_case(&case.id).await.unwrap().unwrap().status,
        "ban_failed"
    );
    assert!(runtime
        .find_active_ban_in_chat(-100, 200)
        .await
        .unwrap()
        .is_none());
    assert!(runtime
        .find_active_bans_for_user(200)
        .await
        .unwrap()
        .is_empty());
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_none());
    let requests = telegram.requests.lock().unwrap();
    assert!(!requests
        .iter()
        .any(|(m, args)| m == "banchatmember" && args["chat_id"] == -300));
    assert!(requests
        .iter()
        .any(|(_, args)| args["text"].as_str().unwrap_or("").contains("封禁失敗")));
}

#[tokio::test]
async fn auto_ban_tracks_partial_success_and_missing_log_without_fake_links() {
    let runtime = test_runtime().await;
    let telegram = TelegramStub::with_failures(
        vec![],
        vec![
            ("deletemessage".to_string(), -100),
            ("sendmessage".to_string(), -1),
        ],
    );
    let mut case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    case.source_message_id = Some(1);
    assert!(
        execute_auto_ban(&telegram.bot, &runtime, case.clone(), "<b>自動封禁</b>")
            .await
            .unwrap()
    );
    let stored = runtime.load_case(&case.id).await.unwrap().unwrap();
    assert_eq!(stored.status, "banned_delete_failed");
    assert!(stored.log_message_id.is_none());
    assert!(runtime
        .find_active_ban_in_chat(-100, 200)
        .await
        .unwrap()
        .is_some());
    assert!(!telegram
        .requests
        .lock()
        .unwrap()
        .iter()
        .any(|(m, args)| m == "sendmessage" && args["chat_id"] == -100));
}

#[tokio::test]
async fn auto_ban_does_not_contact_telegram_if_intent_cannot_be_saved() {
    let runtime = test_runtime().await;
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_cases BEFORE INSERT ON cases BEGIN SELECT RAISE(ABORT, 'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    assert!(execute_auto_ban(&telegram.bot, &runtime, case, "test")
        .await
        .is_err());
    assert!(telegram.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn partial_reversal_survives_restart_and_retries_only_failed_targets() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    runtime.mark_netban_eligible(&case.id).await.unwrap();
    runtime
        .record_network_ban_target(&case.id, -300)
        .await
        .unwrap();
    train_spam(&runtime, "casino gambling", Some(&case.id))
        .await
        .unwrap();
    let failing = TelegramStub::with_failures(vec![], vec![("unbanchatmember".to_string(), -300)]);
    assert!(
        reverse_ban_case(&failing.bot, &runtime, case.clone(), HOST_ID, "Host")
            .await
            .is_err()
    );
    assert_eq!(
        runtime.list_network_ban_targets(&case.id).await.unwrap(),
        vec![-300]
    );
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_none());
    assert!(runtime
        .find_active_ban_in_chat(-100, 200)
        .await
        .unwrap()
        .is_none());
    let reloaded = Runtime::load(runtime.config.clone()).await.unwrap();
    assert_eq!(
        reloaded.load_case(&case.id).await.unwrap().unwrap().status,
        "reversal_pending"
    );
    let success = TelegramStub::new(vec![]);
    reverse_ban_case(&success.bot, &reloaded, case.clone(), HOST_ID, "Host")
        .await
        .unwrap();
    assert_eq!(
        reloaded.load_case(&case.id).await.unwrap().unwrap().status,
        "reversed"
    );
    assert!(reloaded
        .list_network_ban_targets(&case.id)
        .await
        .unwrap()
        .is_empty());
    assert_eq!(reloaded.model.lock().await.spam_docs, 0);
    let requests = success.requests.lock().unwrap();
    let unbans: Vec<_> = requests
        .iter()
        .filter(|(m, _)| m == "unbanchatmember")
        .collect();
    assert_eq!(unbans.len(), 1);
    assert_eq!(unbans[0].1["chat_id"], -300);
    assert_eq!(unbans[0].1["only_if_banned"], true);
}

#[tokio::test]
async fn reversal_preserves_an_independent_ban_in_the_same_chat() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    let other = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    runtime.persist_case(&other).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    reverse_ban_case(&telegram.bot, &runtime, case.clone(), HOST_ID, "Host")
        .await
        .unwrap();
    assert_eq!(
        runtime
            .find_active_ban_in_chat(-100, 200)
            .await
            .unwrap()
            .unwrap()
            .id,
        other.id
    );
    assert!(!telegram
        .requests
        .lock()
        .unwrap()
        .iter()
        .any(|(m, _)| m == "unbanchatmember"));
}

#[tokio::test]
async fn reliability_migration_is_additive_and_idempotent() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    runtime
        .with_conn(|conn| {
            conn.execute_batch("DROP TABLE training_reviews; PRAGMA user_version=17;")?;
            Runtime::init_db(conn)?;
            Runtime::init_db(conn)?;
            assert_eq!(
                conn.query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))?,
                29
            );
            Ok(())
        })
        .await
        .unwrap();
    assert!(runtime.load_case(&case.id).await.unwrap().is_some());
    assert!(runtime
        .decide_training_review(&case.id, "reject", HOST_ID)
        .await
        .unwrap());
}
