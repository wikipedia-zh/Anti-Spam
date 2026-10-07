use super::*;

fn error() -> serde_json::Value {
    serde_json::json!({"ok":false,"error_code":400,"description":"Bad Request: injected failure"})
}
fn api(failures: Vec<(&str, i64)>) -> TelegramStub {
    TelegramStub::with_members(
        vec![555],
        failures
            .into_iter()
            .map(|(method, chat)| (method.into(), chat, error()))
            .collect(),
        true,
    )
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
fn candidate() -> CaseRecord {
    let mut case = dummy_case(ActionKind::Mute, -100, 200, Utc::now());
    case.actor_user_id = Some(555);
    case.actor_name = Some("Admin".into());
    case
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
async fn due(runtime: &Runtime) {
    runtime.with_conn(|conn|{conn.execute_batch("UPDATE restriction_jobs SET next_attempt_at=0;UPDATE origin_ban_jobs SET next_attempt_at=0;UPDATE warning_requests SET next_attempt_at=0;UPDATE telegram_retry_state SET not_before=0;")?;Ok(())}).await.unwrap();
}
async fn command(runtime: &Runtime, api: &TelegramStub, ot: bool) {
    crate::warning_queue::handle(
        &api.bot,
        runtime,
        &spam_ban_message(555, if ot { "/ot" } else { "/warn" }, Some("evidence")),
        200,
        "Member".into(),
        "reason".into(),
        ot.then_some(1),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn warning_count_and_escalation_commit_together_and_replays_do_not_add_warns() {
    let runtime = test_runtime().await;
    runtime.set_ot_warn_count(-100, 3).await.unwrap();
    let (a, b) = tokio::join!(
        runtime.queue_warning(candidate(), 10, "off topic".into(), Some(1)),
        runtime.queue_warning(candidate(), 10, "off topic".into(), Some(1))
    );
    assert_eq!(
        serde_json::to_value(a.unwrap()).unwrap(),
        serde_json::to_value(b.unwrap()).unwrap()
    );
    assert_eq!(runtime.warn_count(-100, 200).await.unwrap(), 3);
    assert_eq!(only_case(&runtime).await.status, "action_pending");
    runtime.with_conn(|conn| {
        assert_eq!(conn.query_row("SELECT COUNT(*) FROM restriction_jobs",[],|r|r.get::<_,i64>(0))?,1);
        assert_eq!(conn.query_row("SELECT COUNT(*) FROM warning_requests",[],|r|r.get::<_,i64>(0))?,1);
        conn.execute_batch("CREATE TRIGGER fail_warning BEFORE INSERT ON warning_requests BEGIN SELECT RAISE(ABORT,'injected'); END;")?;Ok(())
    }).await.unwrap();
    assert!(runtime
        .queue_warning(candidate(), 11, "off topic".into(), Some(1))
        .await
        .is_err());
    assert_eq!(runtime.warn_count(-100, 200).await.unwrap(), 3);
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM cases", [], |r| r.get::<_, i64>(0))?,
                1
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn failed_warning_notice_does_not_prevent_mute_and_resumes_without_recounting() {
    let runtime = test_runtime().await;
    runtime
        .set_warn_config(-100, 1, "mute", Some(3600))
        .await
        .unwrap();
    let failed = api(vec![("sendmessage", -100)]);
    command(&runtime, &failed, false).await;
    command(&runtime, &failed, false).await;
    assert_eq!(calls(&failed, "restrictchatmember").len(), 1);
    assert_eq!(runtime.warn_count(-100, 200).await.unwrap(), 1);
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let success = api(vec![]);
    due(&restarted).await;
    crate::restriction_retry::retry(&success.bot, &restarted)
        .await
        .unwrap();
    crate::warning_queue::retry(&success.bot, &restarted)
        .await
        .unwrap();
    assert!(calls(&success, "restrictchatmember").is_empty());
    assert!(calls(&success, "sendmessage")
        .iter()
        .any(|v| v["text"].as_str().unwrap_or("").contains("目前累計 1 次")));
    assert_eq!(restarted.warn_count(-100, 200).await.unwrap(), 1);
    assert_eq!(only_case(&restarted).await.status, "done");
}

#[tokio::test]
async fn warning_ban_retries_for_group_admin_without_training_or_a_training_review() {
    let runtime = test_runtime().await;
    runtime.set_warn_config(-100, 1, "ban", None).await.unwrap();
    let failed = api(vec![("banchatmember", -100)]);
    command(&runtime, &failed, false).await;
    assert_eq!(only_case(&runtime).await.status, "ban_failed");
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let success = api(vec![]);
    due(&restarted).await;
    crate::origin_retry::retry_origin_bans(&success.bot, &restarted)
        .await
        .unwrap();
    assert_eq!(calls(&success, "banchatmember").len(), 1);
    assert_eq!(only_case(&restarted).await.status, "done");
    assert_eq!(restarted.model.lock().await.spam_docs, 0);
    restarted
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM ban_followups", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn withdrawn_pending_warning_cannot_be_revived_by_a_later_warning_or_replay() {
    for action in ["mute", "ban"] {
        let runtime = test_runtime().await;
        runtime
            .set_warn_config(-100, 2, action, Some(3600))
            .await
            .unwrap();
        runtime
            .queue_warning(candidate(), 10, "one".into(), None)
            .await
            .unwrap();
        runtime
            .queue_warning(candidate(), 11, "two".into(), None)
            .await
            .unwrap();
        let old = only_case(&runtime).await;
        assert_eq!(
            runtime
                .remove_warning_request(-100, 200, 1, 12)
                .await
                .unwrap(),
            (1, 1)
        );
        assert_eq!(
            runtime
                .remove_warning_request(-100, 200, 1, 12)
                .await
                .unwrap(),
            (1, 1)
        );
        runtime
            .queue_warning(candidate(), 11, "replay".into(), None)
            .await
            .unwrap();
        assert_eq!(runtime.warn_count(-100, 200).await.unwrap(), 1);
        runtime
            .queue_warning(candidate(), 13, "new warning".into(), None)
            .await
            .unwrap();
        let success = api(vec![]);
        due(&runtime).await;
        crate::restriction_retry::retry(&success.bot, &runtime)
            .await
            .unwrap();
        crate::origin_retry::retry_origin_bans(&success.bot, &runtime)
            .await
            .unwrap();
        assert_eq!(
            runtime.load_case(&old.id).await.unwrap().unwrap().status,
            "action_cancelled"
        );
        assert_eq!(
            calls(
                &success,
                if action == "mute" {
                    "restrictchatmember"
                } else {
                    "banchatmember"
                }
            )
            .len(),
            1
        );
        assert_eq!(runtime.warn_count(-100, 200).await.unwrap(), 2);
    }
}

#[tokio::test]
async fn pending_warning_checks_current_policy_and_admin_rights() {
    for action in ["mute", "ban"] {
        for revoked in [false, true] {
            let runtime = test_runtime().await;
            runtime
                .set_warn_config(-100, 1, action, Some(3600))
                .await
                .unwrap();
            runtime
                .queue_warning(candidate(), 10, "reason".into(), None)
                .await
                .unwrap();
            if !revoked {
                runtime
                    .set_warn_config(-100, 5, action, Some(3600))
                    .await
                    .unwrap();
            }
            let success = if revoked {
                TelegramStub::with_members(vec![], vec![], true)
            } else {
                api(vec![])
            };
            crate::restriction_retry::retry(&success.bot, &runtime)
                .await
                .unwrap();
            crate::origin_retry::retry_origin_bans(&success.bot, &runtime)
                .await
                .unwrap();
            assert!(calls(&success, "restrictchatmember").is_empty());
            assert!(calls(&success, "banchatmember").is_empty());
        }
    }
}

#[tokio::test]
async fn ot_deletion_notice_and_expiry_survive_restarts_and_keep_the_saved_template() {
    let runtime = test_runtime().await;
    runtime
        .set_warn_config(-100, 10, "mute", Some(3600))
        .await
        .unwrap();
    runtime
        .set_ot_template(
            -100,
            Some("{user} original {count} {button:Rules}[https://example.com/rules]"),
        )
        .await
        .unwrap();
    let failed = api(vec![("deletemessage", -100)]);
    command(&runtime, &failed, true).await;
    assert!(calls(&failed, "sendmessage").is_empty());
    runtime
        .set_ot_template(-100, Some("changed"))
        .await
        .unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let success = api(vec![]);
    due(&restarted).await;
    crate::warning_queue::retry(&success.bot, &restarted)
        .await
        .unwrap();
    let notices = calls(&success, "sendmessage");
    assert_eq!(notices.len(), 1);
    assert!(notices[0]["text"].as_str().unwrap().contains("original 1"));
    assert_eq!(
        notices[0]["reply_markup"]["inline_keyboard"][0][0]["text"],
        "Rules"
    );
    let snapshot = restarted
        .with_conn(|conn| {
            Ok(
                conn.query_row("SELECT payload FROM warning_requests", [], |r| {
                    r.get::<_, String>(0)
                })?,
            )
        })
        .await
        .unwrap();
    let value: serde_json::Value = serde_json::from_str(&snapshot).unwrap();
    assert_eq!(value["step"], "cleanup");
    assert!(value["cleanup_at"].as_i64().unwrap() > Utc::now().timestamp() + 86300);
    restarted.with_conn(|conn|{conn.execute_batch("UPDATE warning_requests SET payload=json_set(payload,'$.cleanup_at',0),next_attempt_at=0;")?;Ok(())}).await.unwrap();
    let cleanup = api(vec![]);
    crate::warning_queue::retry(&cleanup.bot, &restarted)
        .await
        .unwrap();
    assert_eq!(calls(&cleanup, "deletemessage").len(), 1);
    assert!(calls(&cleanup, "sendmessage").is_empty());
    assert_eq!(restarted.warn_count(-100, 200).await.unwrap(), 1);
}

#[tokio::test]
async fn warning_kick_release_resumes_after_restart_without_kicking_twice() {
    let runtime = test_runtime().await;
    runtime
        .set_warn_config(-100, 1, "kick", None)
        .await
        .unwrap();
    let failed = api(vec![("unbanchatmember", -100)]);
    command(&runtime, &failed, false).await;
    assert_eq!(only_case(&runtime).await.status, "kick_release_pending");
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let success = api(vec![]);
    *success.members.lock().unwrap() = failed.members.lock().unwrap().clone();
    due(&restarted).await;
    crate::restriction_retry::retry(&success.bot, &restarted)
        .await
        .unwrap();
    assert!(calls(&success, "banchatmember").is_empty());
    assert_eq!(calls(&success, "unbanchatmember").len(), 1);
    assert_eq!(only_case(&restarted).await.status, "done");
}

#[tokio::test]
async fn warning_rate_limit_blocks_the_shared_queue_after_restart() {
    let runtime = test_runtime().await;
    let limited = TelegramStub::with_api_errors(
        vec![555],
        vec![(
            "sendmessage".into(),
            -100,
            serde_json::json!({"ok":false,"error_code":429,"description":"Too Many Requests","parameters":{"retry_after":120}}),
        )],
    );
    command(&runtime, &limited, false).await;
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    restarted
        .with_conn(|conn| {
            conn.execute("UPDATE warning_requests SET next_attempt_at=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
    let success = api(vec![]);
    crate::warning_queue::retry(&success.bot, &restarted)
        .await
        .unwrap();
    assert!(calls(&success, "sendmessage").is_empty());
    let queue = restarted.queue_status(None).await.unwrap();
    assert!(queue.contains("警告通知"));
    assert!(queue.contains("Telegram 限流"));
    due(&restarted).await;
    crate::warning_queue::retry(&success.bot, &restarted)
        .await
        .unwrap();
    assert_eq!(calls(&success, "sendmessage").len(), 1);
}

#[tokio::test]
async fn lost_notice_acknowledgement_does_not_add_warnings_or_repeat_the_mute() {
    let runtime = test_runtime().await;
    runtime
        .set_warn_config(-100, 1, "mute", Some(3600))
        .await
        .unwrap();
    runtime.with_conn(|conn| {conn.execute_batch("CREATE TRIGGER fail_warning_ack BEFORE UPDATE OF payload ON warning_requests WHEN json_extract(NEW.payload,'$.step')='command' BEGIN SELECT RAISE(ABORT,'injected'); END;")?;Ok(())}).await.unwrap();
    let telegram = api(vec![]);
    command(&runtime, &telegram, false).await;
    assert_eq!(calls(&telegram, "restrictchatmember").len(), 1);
    runtime
        .with_conn(|conn| {
            conn.execute_batch("DROP TRIGGER fail_warning_ack;")?;
            Ok(())
        })
        .await
        .unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due(&restarted).await;
    command(&restarted, &telegram, false).await;
    assert_eq!(restarted.warn_count(-100, 200).await.unwrap(), 1);
    assert_eq!(calls(&telegram, "restrictchatmember").len(), 1);
    // Telegram sends have no idempotency key; a lost ack may duplicate only the notice.
    assert_eq!(
        calls(&telegram, "sendmessage")
            .iter()
            .filter(|v| v["text"].as_str().unwrap_or("").contains("目前累計 1 次"))
            .count(),
        2
    );
}

#[tokio::test]
async fn reversing_a_warning_mute_does_not_allow_command_replay_to_apply_it_again() {
    let runtime = test_runtime().await;
    runtime
        .set_warn_config(-100, 1, "mute", Some(3600))
        .await
        .unwrap();
    let telegram = api(vec![]);
    command(&runtime, &telegram, false).await;
    let case = only_case(&runtime).await;
    crate::restriction_retry::reverse(&telegram.bot, &runtime, case, 555, "Admin")
        .await
        .unwrap();
    due(&runtime).await;
    command(&runtime, &telegram, false).await;
    assert_eq!(runtime.warn_count(-100, 200).await.unwrap(), 1);
    assert_eq!(only_case(&runtime).await.action, ActionKind::Unmuted);
    assert_eq!(calls(&telegram, "restrictchatmember").len(), 2);
}
