use super::*;

async fn due_now(runtime: &Runtime) {
    runtime.with_conn(|conn| {
        conn.execute_batch("UPDATE origin_ban_jobs SET next_attempt_at=0; UPDATE review_updates SET next_attempt_at=0; UPDATE telegram_retry_state SET not_before=0;")?;
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
        .map(|(_, a)| a.clone())
        .collect()
}

async fn only_case(runtime: &Runtime) -> CaseRecord {
    let id = runtime
        .with_conn(
            |conn| Ok(conn.query_row("SELECT id FROM cases", [], |r| r.get::<_, String>(0))?),
        )
        .await
        .unwrap();
    runtime.load_case(&id).await.unwrap().unwrap()
}

async fn report(runtime: &Runtime) -> CaseRecord {
    let mut case = dummy_case(ActionKind::PendingReport, -100, 200, Utc::now());
    case.status = "pending_review".into();
    case.actor_user_id = Some(444);
    case.source_message_id = Some(1);
    case.evidence_text = "casino gambling".into();
    runtime.persist_case(&case).await.unwrap();
    case
}

fn callback(case: &CaseRecord, decision: &str) -> CallbackQuery {
    serde_json::from_value(serde_json::json!({
        "id":Uuid::new_v4().to_string(),"chat_instance":"review",
        "from":{"id":HOST_ID,"is_bot":false,"first_name":"Host"},
        "data":format!("review:{decision}:{}",case.id),
        "message":{"message_id":10,"date":1,"text":"Review","chat":{"id":-1,"type":"channel","title":"Reviews"}}
    })).unwrap()
}

#[tokio::test]
async fn training_review_notification_retries_without_training_twice() {
    let runtime = Arc::new(test_runtime().await);
    let mut case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    case.evidence_text = "casino gambling".into();
    runtime.persist_case(&case).await.unwrap();
    let mut review = callback(&case, "approve");
    review.data = Some(format!("train:approve:{}", case.id));
    let failed = TelegramStub::with_failures(vec![], vec![("editmessagetext".into(), -1)]);
    handle_callback(failed.bot.clone(), runtime.clone(), review)
        .await
        .unwrap();
    assert_eq!(runtime.model.lock().await.spam_docs, 1);
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due_now(&restarted).await;
    let success = TelegramStub::new(vec![]);
    crate::moderation_queue::deliver_review_updates(&success.bot, &restarted, None)
        .await
        .unwrap();
    let edits = calls(&success, "editmessagetext");
    assert_eq!(edits.len(), 1);
    assert!(edits[0]["text"]
        .as_str()
        .unwrap()
        .contains("已訓練並加入跨群黑名單"));
    assert!(edits[0]["text"]
        .as_str()
        .unwrap()
        .contains(&HOST_ID.to_string()));
    assert_eq!(
        edits[0]["reply_markup"]["inline_keyboard"],
        serde_json::json!([])
    );
    assert_eq!(restarted.model.lock().await.spam_docs, 1);
    due_now(&restarted).await;
    crate::moderation_queue::deliver_review_updates(&success.bot, &restarted, None)
        .await
        .unwrap();
    assert_eq!(calls(&success, "editmessagetext").len(), 1);
}

#[tokio::test]
async fn review_rate_limit_delays_both_messages_after_restart() {
    let runtime = test_runtime().await;
    let case = report(&runtime).await;
    runtime
        .set_report_confirmation(&case.id, -100, 42)
        .await
        .unwrap();
    let guard = runtime.review_guard(&case.id).await;
    runtime
        .decide_report(&case, "reject", (HOST_ID, "Host".into()), (-1, 10), guard)
        .await
        .unwrap();
    let limited = TelegramStub::with_api_errors(
        vec![],
        vec![(
            "editmessagetext".into(),
            -1,
            serde_json::json!({"ok":false,"error_code":429,"description":"Too Many Requests","parameters":{"retry_after":120}}),
        )],
    );
    crate::moderation_queue::deliver_review_updates(&limited.bot, &runtime, None)
        .await
        .unwrap();
    assert_eq!(calls(&limited, "editmessagetext").len(), 1);
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    restarted
        .with_conn(|conn| {
            conn.execute("UPDATE review_updates SET next_attempt_at=0", [])?;
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
    crate::moderation_queue::deliver_review_updates(&success.bot, &restarted, None)
        .await
        .unwrap();
    assert!(calls(&success, "editmessagetext").is_empty());
    due_now(&restarted).await;
    crate::moderation_queue::deliver_review_updates(&success.bot, &restarted, None)
        .await
        .unwrap();
    assert_eq!(calls(&success, "editmessagetext").len(), 2);
}

#[tokio::test]
async fn failed_bot_rule_write_retries_without_rebanning_or_training_handles() {
    let runtime = Arc::new(test_runtime().await);
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_rule BEFORE INSERT ON spam_rules BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    handle_ban_mute_kick(
        telegram.bot.clone(),
        runtime.clone(),
        spam_ban_message(HOST_ID, "/sb -f", Some("@CasinoBot")),
        parse_command("/sb -f"),
    )
    .await
    .unwrap();
    assert_eq!(calls(&telegram, "banchatmember").len(), 1);
    assert_eq!(runtime.model.lock().await.spam_docs, 0);
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT state FROM origin_ban_jobs", [], |r| r
                    .get::<_, String>(0))?,
                "pending"
            );
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM spam_rules", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            conn.execute_batch("DROP TRIGGER fail_rule;")?;
            Ok(())
        })
        .await
        .unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due_now(&restarted).await;
    let success = TelegramStub::new(vec![]);
    crate::origin_retry::retry_origin_bans(&success.bot, &restarted)
        .await
        .unwrap();
    assert!(calls(&success, "banchatmember").is_empty());
    assert_eq!(restarted.model.lock().await.spam_docs, 0);
    assert_eq!(restarted.spam_rules.read().await.len(), 1);
    restarted
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT state FROM origin_ban_jobs", [], |r| r
                    .get::<_, String>(0))?,
                "done"
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn manual_bans_retry_after_restart_without_replaying_commands_or_training_early() {
    for force in [false, true] {
        let runtime = Arc::new(test_runtime().await);
        runtime
            .set_group_module(-300, "netban", true)
            .await
            .unwrap();
        let actor = if force { HOST_ID } else { 555 };
        let command = if force { "/sb -f" } else { "/sb" };
        let message = spam_ban_message(actor, command, Some("casino gambling"));
        let failed = TelegramStub::with_failures(vec![555], vec![("banchatmember".into(), -100)]);
        handle_ban_mute_kick(
            failed.bot.clone(),
            runtime.clone(),
            message.clone(),
            parse_command(command),
        )
        .await
        .unwrap();
        let case = only_case(&runtime).await;
        assert_eq!(case.status, "ban_failed");
        assert_eq!(runtime.model.lock().await.spam_docs, 0);
        assert!(runtime
            .find_active_network_ban(200)
            .await
            .unwrap()
            .is_none());
        assert!(!calls(&failed, "deletemessage")
            .iter()
            .any(|a| a["message_id"] == 1));
        handle_ban_mute_kick(
            failed.bot.clone(),
            runtime.clone(),
            message,
            parse_command(command),
        )
        .await
        .unwrap();
        assert_eq!(calls(&failed, "banchatmember").len(), 1);
        runtime
            .with_conn(|conn| {
                assert_eq!(
                    conn.query_row("SELECT COUNT(*) FROM cases", [], |r| r.get::<_, i64>(0))?,
                    1
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
        let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
        due_now(&restarted).await;
        let success = TelegramStub::new(vec![555]);
        crate::origin_retry::retry_origin_bans(&success.bot, &restarted)
            .await
            .unwrap();
        deliver_network_bans(&success.bot, &restarted, None)
            .await
            .unwrap();
        assert_eq!(restarted.model.lock().await.spam_docs, u64::from(force));
        assert_eq!(
            restarted
                .find_active_network_ban(200)
                .await
                .unwrap()
                .is_some(),
            force
        );
        assert_eq!(
            restarted.load_case(&case.id).await.unwrap().unwrap().status,
            if force { "force_approved" } else { "done" }
        );
        assert_eq!(
            calls(&success, "sendmessage")
                .iter()
                .any(|a| a.to_string().contains("train:approve:")),
            !force
        );
    }
}

#[tokio::test]
async fn training_failure_keeps_the_ban_and_retries_the_sample_and_network_together() {
    let runtime = Arc::new(test_runtime().await);
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_training BEFORE INSERT ON word_frequencies BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    handle_ban_mute_kick(
        telegram.bot.clone(),
        runtime.clone(),
        spam_ban_message(HOST_ID, "/sb -f", Some("casino gambling")),
        parse_command("/sb -f"),
    )
    .await
    .unwrap();
    assert!(runtime
        .find_active_ban_in_chat(-100, 200)
        .await
        .unwrap()
        .is_some());
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_none());
    assert_eq!(runtime.model.lock().await.spam_docs, 0);
    assert_eq!(calls(&telegram, "banchatmember").len(), 1);
    runtime
        .with_conn(|conn| {
            conn.execute_batch("DROP TRIGGER fail_training;")?;
            Ok(())
        })
        .await
        .unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due_now(&restarted).await;
    let success = TelegramStub::new(vec![]);
    crate::origin_retry::retry_origin_bans(&success.bot, &restarted)
        .await
        .unwrap();
    assert!(calls(&success, "banchatmember").is_empty());
    assert!(calls(&success, "deletemessage").is_empty());
    deliver_network_bans(&success.bot, &restarted, None)
        .await
        .unwrap();
    assert_eq!(calls(&success, "banchatmember").len(), 1);
    assert_eq!(restarted.model.lock().await.spam_docs, 1);
    due_now(&restarted).await;
    crate::origin_retry::retry_origin_bans(&success.bot, &restarted)
        .await
        .unwrap();
    assert_eq!(restarted.model.lock().await.spam_docs, 1);
}

#[tokio::test]
async fn failed_manual_audit_write_rolls_back_the_entire_intent() {
    let runtime = Arc::new(test_runtime().await);
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_audit BEFORE INSERT ON maintainer_actions BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    handle_ban_mute_kick(
        telegram.bot.clone(),
        runtime.clone(),
        spam_ban_message(HOST_ID, "/sb -f", Some("casino gambling")),
        parse_command("/sb -f"),
    )
    .await
    .unwrap();
    assert!(calls(&telegram, "banchatmember").is_empty());
    assert!(calls(&telegram, "deletemessage").is_empty());
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM cases", [], |r| r.get::<_, i64>(0))?,
                0
            );
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM moderation_requests", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn report_approval_retries_the_ban_without_losing_its_decision_or_confirmations() {
    let runtime = Arc::new(test_runtime().await);
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    let case = report(&runtime).await;
    runtime
        .set_report_confirmation(&case.id, -100, 42)
        .await
        .unwrap();
    let failed = TelegramStub::with_failures(vec![], vec![("banchatmember".into(), -100)]);
    handle_callback(
        failed.bot.clone(),
        runtime.clone(),
        callback(&case, "approve"),
    )
    .await
    .unwrap();
    let stored = runtime.load_case(&case.id).await.unwrap().unwrap();
    assert_eq!(stored.action, ActionKind::ReportApproved);
    assert_eq!(stored.status, "ban_failed");
    assert_eq!(runtime.model.lock().await.spam_docs, 0);
    assert!(!calls(&failed, "editmessagetext")
        .iter()
        .any(|a| a["text"].as_str().unwrap_or("").contains("已受理並封禁")));
    handle_callback(
        failed.bot.clone(),
        runtime.clone(),
        callback(&case, "reject"),
    )
    .await
    .unwrap();
    assert_eq!(runtime.report_strikes(444).await, 0);
    let restarted = Arc::new(Runtime::load(runtime.config.clone()).await.unwrap());
    due_now(&restarted).await;
    let success = TelegramStub::new(vec![]);
    crate::origin_retry::retry_origin_bans(&success.bot, &restarted)
        .await
        .unwrap();
    crate::moderation_queue::deliver_review_updates(&success.bot, &restarted, None)
        .await
        .unwrap();
    deliver_network_bans(&success.bot, &restarted, None)
        .await
        .unwrap();
    assert_eq!(restarted.model.lock().await.spam_docs, 1);
    assert!(restarted
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_some());
    assert_eq!(
        calls(&success, "editmessagetext").len(),
        3,
        "update the case log, review card and reporter confirmation"
    );
    assert!(calls(&success, "editmessagetext")
        .iter()
        .any(|a| a["message_id"] == 42 && a["text"].as_str().unwrap().contains("已被封禁")));
    handle_callback(
        success.bot.clone(),
        restarted.clone(),
        callback(&case, "approve"),
    )
    .await
    .unwrap();
    assert_eq!(restarted.model.lock().await.spam_docs, 1);
}

#[tokio::test]
async fn rejection_rolls_back_training_strikes_and_notification_intent_together() {
    let runtime = Arc::new(test_runtime().await);
    let case = report(&runtime).await;
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_reject BEFORE UPDATE OF action ON cases WHEN NEW.action='report_rejected' BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    handle_callback(
        telegram.bot.clone(),
        runtime.clone(),
        callback(&case, "reject"),
    )
    .await
    .unwrap();
    assert_eq!(runtime.model.lock().await.ham_docs, 0);
    assert_eq!(runtime.report_strikes(444).await, 0);
    assert_eq!(
        runtime.load_case(&case.id).await.unwrap().unwrap().status,
        "pending_review"
    );
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM review_updates", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            conn.execute_batch("DROP TRIGGER fail_reject;")?;
            Ok(())
        })
        .await
        .unwrap();
    let (a, b) = tokio::join!(
        handle_callback(
            telegram.bot.clone(),
            runtime.clone(),
            callback(&case, "reject")
        ),
        handle_callback(
            telegram.bot.clone(),
            runtime.clone(),
            callback(&case, "reject")
        )
    );
    a.unwrap();
    b.unwrap();
    assert_eq!(runtime.model.lock().await.ham_docs, 1);
    assert_eq!(runtime.report_strikes(444).await, 1);
}

#[tokio::test]
async fn failed_review_updates_survive_restart_and_late_confirmation_messages() {
    let runtime = Arc::new(test_runtime().await);
    let case = report(&runtime).await;
    let failed = TelegramStub::with_failures(vec![], vec![("editmessagetext".into(), -1)]);
    handle_callback(
        failed.bot.clone(),
        runtime.clone(),
        callback(&case, "reject"),
    )
    .await
    .unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due_now(&restarted).await;
    let success = TelegramStub::new(vec![]);
    crate::moderation_queue::deliver_review_updates(&success.bot, &restarted, None)
        .await
        .unwrap();
    assert_eq!(calls(&success, "editmessagetext").len(), 1);
    restarted
        .set_report_confirmation(&case.id, -100, 42)
        .await
        .unwrap();
    crate::moderation_queue::deliver_review_updates(&success.bot, &restarted, None)
        .await
        .unwrap();
    assert_eq!(calls(&success, "editmessagetext").len(), 2);
    assert_eq!(calls(&success, "editmessagetext")[1]["message_id"], 42);
    due_now(&restarted).await;
    crate::moderation_queue::deliver_review_updates(&success.bot, &restarted, None)
        .await
        .unwrap();
    assert_eq!(calls(&success, "editmessagetext").len(), 2);
}

#[tokio::test]
async fn reversing_an_approved_report_cancels_ban_training_and_updates_the_review() {
    let runtime = Arc::new(test_runtime().await);
    let case = report(&runtime).await;
    let guard = runtime.review_guard(&case.id).await;
    runtime
        .decide_report(&case, "approve", (HOST_ID, "Host".into()), (-1, 10), guard)
        .await
        .unwrap();
    let telegram = TelegramStub::new(vec![]);
    reverse_ban_case(&telegram.bot, &runtime, case.clone(), HOST_ID, "Host")
        .await
        .unwrap();
    due_now(&runtime).await;
    crate::origin_retry::retry_origin_bans(&telegram.bot, &runtime)
        .await
        .unwrap();
    crate::moderation_queue::deliver_review_updates(&telegram.bot, &runtime, None)
        .await
        .unwrap();
    assert!(calls(&telegram, "banchatmember").is_empty());
    assert_eq!(runtime.model.lock().await.spam_docs, 0);
    assert!(calls(&telegram, "editmessagetext")
        .iter()
        .any(|a| a["text"].as_str().unwrap_or("").contains("已撤銷")));
}

#[tokio::test]
async fn a_revoked_reviewer_cannot_resume_a_pending_forced_ban() {
    let runtime = Arc::new(test_runtime().await);
    runtime
        .set_reviewer(555, true, Some(HOST_ID))
        .await
        .unwrap();
    runtime.delay_telegram_queue(300).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    handle_ban_mute_kick(
        telegram.bot.clone(),
        runtime.clone(),
        spam_ban_message(555, "/sb -f", Some("casino gambling")),
        parse_command("/sb -f"),
    )
    .await
    .unwrap();
    runtime
        .set_reviewer(555, false, Some(HOST_ID))
        .await
        .unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due_now(&restarted).await;
    crate::origin_retry::retry_origin_bans(&telegram.bot, &restarted)
        .await
        .unwrap();
    assert!(calls(&telegram, "banchatmember").is_empty());
    assert_eq!(restarted.model.lock().await.spam_docs, 0);
    restarted
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT state FROM origin_ban_jobs", [], |r| r
                    .get::<_, String>(0))?,
                "cancelled"
            );
            Ok(())
        })
        .await
        .unwrap();
}
