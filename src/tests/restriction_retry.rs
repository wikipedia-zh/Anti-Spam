use super::*;

fn api(errors: Vec<(&str, serde_json::Value)>) -> TelegramStub {
    TelegramStub::with_members(
        vec![555],
        errors
            .into_iter()
            .map(|(m, e)| (m.into(), -100, e))
            .collect(),
        true,
    )
}
fn error() -> serde_json::Value {
    serde_json::json!({"ok":false,"error_code":400,"description":"Bad Request: injected failure"})
}
fn calls(api: &TelegramStub, method: &str) -> Vec<serde_json::Value> {
    api.requests
        .lock()
        .unwrap()
        .iter()
        .filter(|(m, _)| m == method)
        .map(|(_, v)| v.clone())
        .collect()
}
async fn due(runtime: &Runtime) {
    runtime.with_conn(|conn|{conn.execute_batch("UPDATE restriction_jobs SET next_attempt_at=0;UPDATE telegram_retry_state SET not_before=0;")?;Ok(())}).await.unwrap();
}
async fn case(runtime: &Runtime) -> CaseRecord {
    let id = runtime
        .with_conn(|conn| {
            Ok(conn.query_row(
                "SELECT id FROM cases ORDER BY created_at LIMIT 1",
                [],
                |r| r.get::<_, String>(0),
            )?)
        })
        .await
        .unwrap();
    runtime.load_case(&id).await.unwrap().unwrap()
}
async fn command(runtime: Arc<Runtime>, api: &TelegramStub, text: &str) {
    handle_ban_mute_kick(
        api.bot.clone(),
        runtime,
        spam_ban_message(555, text, Some("message")),
        parse_command(text),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn failed_mute_is_persisted_and_deduplicated_then_resumes_after_restart() {
    let runtime = Arc::new(test_runtime().await);
    let failed = api(vec![("restrictchatmember", error())]);
    command(runtime.clone(), &failed, "/mute").await;
    assert_eq!(case(&runtime).await.status, "action_failed");
    assert!(!calls(&failed, "sendmessage")
        .iter()
        .any(|v| v["text"].as_str().unwrap_or("").contains("<b>禁言</b>")));
    command(runtime.clone(), &failed, "/mute").await;
    assert_eq!(calls(&failed, "restrictchatmember").len(), 1);
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let success = api(vec![]);
    due(&restarted).await;
    crate::restriction_retry::retry(&success.bot, &restarted)
        .await
        .unwrap();
    assert_eq!(case(&restarted).await.status, "done");
    assert_eq!(calls(&success, "restrictchatmember").len(), 1);
    due(&restarted).await;
    crate::restriction_retry::retry(&success.bot, &restarted)
        .await
        .unwrap();
    assert_eq!(calls(&success, "restrictchatmember").len(), 1);
}

#[tokio::test]
async fn kick_release_resumes_without_banning_again_or_removing_a_rejoined_user() {
    for rejoined in [false, true] {
        let runtime = Arc::new(test_runtime().await);
        let failed = api(vec![("unbanchatmember", error())]);
        command(runtime.clone(), &failed, "/kick").await;
        assert_eq!(case(&runtime).await.status, "kick_release_pending");
        let ban = calls(&failed, "banchatmember");
        assert_eq!(ban.len(), 1);
        assert!(ban[0]["until_date"].as_i64().unwrap() > Utc::now().timestamp() + 30);
        let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
        let success = api(vec![]);
        if !rejoined {
            *success.members.lock().unwrap() = failed.members.lock().unwrap().clone();
        }
        due(&restarted).await;
        crate::restriction_retry::retry(&success.bot, &restarted)
            .await
            .unwrap();
        assert!(calls(&success, "banchatmember").is_empty());
        assert_eq!(
            calls(&success, "unbanchatmember").len(),
            usize::from(!rejoined)
        );
        if !rejoined {
            assert_eq!(
                calls(&success, "unbanchatmember")[0]["only_if_banned"],
                true
            );
        }
        assert_eq!(case(&restarted).await.status, "done");
    }
}

#[tokio::test]
async fn kick_does_not_release_an_independent_ban() {
    let runtime = Arc::new(test_runtime().await);
    let failed = api(vec![("unbanchatmember", error())]);
    command(runtime.clone(), &failed, "/kick").await;
    let other = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&other).await.unwrap();
    let success = api(vec![]);
    *success.members.lock().unwrap() = failed.members.lock().unwrap().clone();
    due(&runtime).await;
    crate::restriction_retry::retry(&success.bot, &runtime)
        .await
        .unwrap();
    assert!(calls(&success, "unbanchatmember").is_empty());
}

#[tokio::test]
async fn mute_api_success_followed_by_failed_ack_is_reconciled() {
    let runtime = Arc::new(test_runtime().await);
    runtime.with_conn(|conn|{conn.execute_batch("CREATE TRIGGER fail_ack BEFORE UPDATE OF status ON cases WHEN NEW.status='done' BEGIN SELECT RAISE(ABORT,'injected'); END;")?;Ok(())}).await.unwrap();
    let telegram = api(vec![]);
    command(runtime.clone(), &telegram, "/mute").await;
    assert_eq!(calls(&telegram, "restrictchatmember").len(), 1);
    assert_eq!(case(&runtime).await.status, "action_pending");
    runtime
        .with_conn(|conn| {
            conn.execute_batch("DROP TRIGGER fail_ack;")?;
            Ok(())
        })
        .await
        .unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due(&restarted).await;
    crate::restriction_retry::retry(&telegram.bot, &restarted)
        .await
        .unwrap();
    assert_eq!(calls(&telegram, "restrictchatmember").len(), 1);
    assert_eq!(case(&restarted).await.status, "done");
}

#[tokio::test]
async fn reversal_cancels_a_pending_mute_and_failed_unmutes_resume() {
    let runtime = Arc::new(test_runtime().await);
    let failed = api(vec![("restrictchatmember", error())]);
    command(runtime.clone(), &failed, "/mute").await;
    let saved = case(&runtime).await;
    let success = api(vec![]);
    reverse_mute_case(&success.bot, &runtime, saved.clone(), HOST_ID, "Host")
        .await
        .unwrap();
    due(&runtime).await;
    crate::restriction_retry::retry(&success.bot, &runtime)
        .await
        .unwrap();
    assert!(calls(&success, "restrictchatmember").is_empty());
    assert_eq!(case(&runtime).await.status, "reversed");

    let second = Arc::new(test_runtime().await);
    let working = api(vec![]);
    command(second.clone(), &working, "/mute").await;
    let failing = api(vec![("restrictchatmember", error())]);
    *failing.members.lock().unwrap() = working.members.lock().unwrap().clone();
    let response = reverse_mute_case(&failing.bot, &second, case(&second).await, HOST_ID, "Host")
        .await
        .unwrap();
    assert!(response.contains("尚未完成"));
    let restarted = Runtime::load(second.config.clone()).await.unwrap();
    due(&restarted).await;
    crate::restriction_retry::retry(&working.bot, &restarted)
        .await
        .unwrap();
    assert_eq!(case(&restarted).await.status, "reversed");
    assert_eq!(calls(&working, "restrictchatmember").len(), 2);
}

#[tokio::test]
async fn case_reversal_preserves_other_mutes_and_user_unmute_cancels_pending_work() {
    let runtime = Arc::new(test_runtime().await);
    let telegram = api(vec![]);
    command(runtime.clone(), &telegram, "/mute").await;
    let first = case(&runtime).await;
    let later = dummy_case(ActionKind::Mute, -100, 200, Utc::now());
    runtime.persist_case(&later).await.unwrap();
    reverse_mute_case(&telegram.bot, &runtime, first, HOST_ID, "Host")
        .await
        .unwrap();
    assert_eq!(calls(&telegram, "restrictchatmember").len(), 1);
    let pending = dummy_case(ActionKind::FloodMute, -100, 200, Utc::now());
    runtime
        .queue_restriction(pending, 99, None, "mute")
        .await
        .unwrap();
    crate::restriction_retry::release_all(&telegram.bot, &runtime, -100, 200, 555, "Admin", 100)
        .await
        .unwrap();
    due(&runtime).await;
    crate::restriction_retry::retry(&telegram.bot, &runtime)
        .await
        .unwrap();
    assert_eq!(calls(&telegram, "restrictchatmember").len(), 2);
    assert_eq!(
        runtime.load_case(&later.id).await.unwrap().unwrap().status,
        "locally_unmuted"
    );
}

#[tokio::test]
async fn temporary_mutes_keep_their_deadline_and_expired_work_cannot_mute_forever() {
    for expired in [false, true] {
        let runtime = test_runtime().await;
        let until = Utc::now().timestamp() + if expired { 10 } else { 300 };
        let saved = dummy_case(ActionKind::Mute, -100, 200, Utc::now());
        runtime
            .queue_restriction(saved.clone(), 1, Some(until), "mute")
            .await
            .unwrap();
        let failed = api(vec![("restrictchatmember", error())]);
        let _ = crate::restriction_retry::attempt(&failed.bot, &runtime, saved).await;
        let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
        due(&restarted).await;
        let success = api(vec![]);
        crate::restriction_retry::retry(&success.bot, &restarted)
            .await
            .unwrap();
        if expired {
            assert!(calls(&success, "restrictchatmember").is_empty());
            assert_eq!(case(&restarted).await.status, "action_cancelled");
        } else {
            assert_eq!(
                calls(&success, "restrictchatmember")[0]["until_date"],
                until
            );
        }
    }
}

#[tokio::test]
async fn restriction_rate_limit_and_notification_failure_survive_restart() {
    let runtime = Arc::new(test_runtime().await);
    let limited = api(vec![(
        "restrictchatmember",
        serde_json::json!({"ok":false,"error_code":429,"description":"Too Many Requests","parameters":{"retry_after":120}}),
    )]);
    command(runtime.clone(), &limited, "/mute").await;
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    restarted
        .with_conn(|conn| {
            conn.execute("UPDATE restriction_jobs SET next_attempt_at=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
    let failed_notice = api(vec![("sendmessage", error())]);
    crate::restriction_retry::retry(&failed_notice.bot, &restarted)
        .await
        .unwrap();
    assert!(calls(&failed_notice, "restrictchatmember").is_empty());
    due(&restarted).await;
    crate::restriction_retry::retry(&failed_notice.bot, &restarted)
        .await
        .unwrap();
    assert_eq!(calls(&failed_notice, "restrictchatmember").len(), 1);
    assert_eq!(case(&restarted).await.status, "done");
    let again = Runtime::load(runtime.config.clone()).await.unwrap();
    due(&again).await;
    let success = api(vec![]);
    crate::restriction_retry::retry(&success.bot, &again)
        .await
        .unwrap();
    assert!(calls(&success, "restrictchatmember").is_empty());
    assert!(calls(&success, "sendmessage")
        .iter()
        .any(|v| v["chat_id"] == -100));
}

#[tokio::test]
async fn audit_failure_prevents_mute_and_revoked_admin_cannot_resume_it() {
    let runtime = Arc::new(test_runtime().await);
    runtime.with_conn(|conn|{conn.execute_batch("CREATE TRIGGER fail_audit BEFORE INSERT ON maintainer_actions BEGIN SELECT RAISE(ABORT,'injected'); END;")?;Ok(())}).await.unwrap();
    let telegram = api(vec![]);
    command(runtime.clone(), &telegram, "/mute").await;
    assert!(calls(&telegram, "restrictchatmember").is_empty());
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM cases", [], |r| r.get::<_, i64>(0))?,
                0
            );
            conn.execute_batch("DROP TRIGGER fail_audit;")?;
            Ok(())
        })
        .await
        .unwrap();
    runtime.delay_telegram_queue(300).await.unwrap();
    command(runtime.clone(), &telegram, "/mute").await;
    let revoked = TelegramStub::new(vec![]);
    due(&runtime).await;
    crate::restriction_retry::retry(&revoked.bot, &runtime)
        .await
        .unwrap();
    assert!(calls(&revoked, "restrictchatmember").is_empty());
    assert_eq!(case(&runtime).await.status, "action_cancelled");
}

#[tokio::test]
async fn an_unmute_retry_preserves_a_later_external_restriction() {
    let runtime = Arc::new(test_runtime().await);
    let initial = api(vec![]);
    command(runtime.clone(), &initial, "/mute").await;
    let failed = api(vec![("restrictchatmember", error())]);
    *failed.members.lock().unwrap() = initial.members.lock().unwrap().clone();
    crate::restriction_retry::release_all(&failed.bot, &runtime, -100, 200, 555, "Admin", 80)
        .await
        .unwrap();
    let success = api(vec![]);
    *success.members.lock().unwrap() = initial.members.lock().unwrap().clone();
    success
        .members
        .lock()
        .unwrap()
        .get_mut(&(-100, 200))
        .unwrap()["until_date"] = serde_json::json!(Utc::now().timestamp() + 3600);
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due(&restarted).await;
    crate::restriction_retry::retry(&success.bot, &restarted)
        .await
        .unwrap();
    assert!(calls(&success, "restrictchatmember").is_empty());
    assert!(calls(&success, "sendmessage")
        .iter()
        .any(|v| v["text"].as_str().unwrap_or("").contains("其他限制仍保留")));
}
