use super::*;

fn calls(stub: &TelegramStub, method: &str) -> usize {
    stub.requests
        .lock()
        .unwrap()
        .iter()
        .filter(|(m, _)| m == method)
        .count()
}
fn notices(stub: &TelegramStub) -> usize {
    stub.requests
        .lock()
        .unwrap()
        .iter()
        .filter(|(m, a)| {
            m == "sendmessage"
                && a["text"]
                    .as_str()
                    .is_some_and(|s| s.contains("已同步跨群封禁"))
        })
        .count()
}
async fn setup() -> (Runtime, CaseRecord) {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    // Mark eligibility without initially propagating to this group.
    let id = case.id.clone();
    runtime
        .with_conn(move |c| {
            c.execute("UPDATE cases SET netban_eligible=1 WHERE id=?1", [id])?;
            Ok(())
        })
        .await
        .unwrap();
    (runtime, case)
}
async fn due(runtime: &Runtime) {
    runtime.with_conn(|c|{c.execute_batch("UPDATE network_deliveries SET next_attempt_at=0; UPDATE network_catchups SET next_attempt_at=0;")?;Ok(())}).await.unwrap();
}
async fn drain(stub: &TelegramStub, runtime: &Runtime) {
    deliver_network_bans(&stub.bot, runtime, None)
        .await
        .unwrap();
    crate::network_catchup::retry(&stub.bot, runtime)
        .await
        .unwrap();
}
async fn state(runtime: &Runtime) -> (String, bool, Option<i32>) {
    runtime.with_conn(|c|Ok(c.query_row("SELECT state,delete_done,notice_id FROM network_catchups ORDER BY message_id LIMIT 1",[],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?)).await.unwrap()
}

#[tokio::test]
async fn failure_restart_replay_and_notice_cleanup() {
    let (runtime, case) = setup().await;
    assert!(runtime
        .queue_network_catchup(&case.id, -300, 5, 200)
        .await
        .unwrap());
    let bad = TelegramStub::with_failures(vec![], vec![("banchatmember".into(), -300)]);
    drain(&bad, &runtime).await;
    assert_eq!(calls(&bad, "banchatmember"), 1);
    assert_eq!(calls(&bad, "sendmessage"), 0);
    assert_eq!(calls(&bad, "deletemessage"), 0);
    assert!(runtime
        .list_network_ban_targets(&case.id)
        .await
        .unwrap()
        .is_empty());
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    assert!(restarted
        .queue_network_catchup(&case.id, -300, 5, 200)
        .await
        .unwrap());
    let good = TelegramStub::new(vec![]);
    drain(&good, &restarted).await;
    assert_eq!(calls(&good, "banchatmember"), 0);
    due(&restarted).await;
    drain(&good, &restarted).await;
    assert_eq!(calls(&good, "banchatmember"), 1);
    assert_eq!(calls(&good, "sendmessage"), 1);
    assert_eq!(calls(&good, "deletemessage"), 1);
    assert_eq!(state(&restarted).await, ("pending".into(), true, Some(100)));
    assert!(restarted
        .queue_network_catchup(&case.id, -300, 5, 200)
        .await
        .unwrap());
    drain(&good, &restarted).await;
    assert_eq!(calls(&good, "banchatmember"), 1);
    assert_eq!(calls(&good, "sendmessage"), 1);
    restarted
        .with_conn(|c| {
            c.execute_batch("UPDATE network_catchups SET cleanup_at=0,next_attempt_at=0;")?;
            Ok(())
        })
        .await
        .unwrap();
    let again = Runtime::load(runtime.config.clone()).await.unwrap();
    drain(&good, &again).await;
    assert_eq!(calls(&good, "deletemessage"), 2);
    assert_eq!(state(&again).await.0, "done");
    assert!(again
        .queue_network_catchup(&case.id, -300, 6, 200)
        .await
        .unwrap());
    drain(&good, &again).await;
    assert_eq!(
        calls(&good, "banchatmember"),
        2,
        "a new event can detect a ban no longer in force"
    );
}

#[tokio::test]
async fn stage_failures_never_repeat_confirmed_bans_or_deletions() {
    for stage in ["deletemessage", "sendmessage"] {
        let (runtime, case) = setup().await;
        runtime
            .queue_network_catchup(&case.id, -300, 5, 200)
            .await
            .unwrap();
        let bad = TelegramStub::with_failures(vec![], vec![(stage.into(), -300)]);
        drain(&bad, &runtime).await;
        assert_eq!(calls(&bad, "banchatmember"), 1);
        let requests = bad.requests.lock().unwrap().len();
        drain(&bad, &runtime).await;
        assert_eq!(
            bad.requests.lock().unwrap().len(),
            requests,
            "notification failures must retain their backoff"
        );
        let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
        due(&restarted).await;
        let good = TelegramStub::new(vec![]);
        drain(&good, &restarted).await;
        assert_eq!(calls(&good, "banchatmember"), 0);
        assert_eq!(
            calls(&good, "deletemessage"),
            usize::from(stage == "deletemessage")
        );
        assert_eq!(calls(&good, "sendmessage"), 1);
    }
}

#[tokio::test]
async fn reversal_cancels_catchups_and_replayed_events_cannot_revive_them() {
    let (runtime, case) = setup().await;
    runtime
        .queue_network_catchup(&case.id, -300, 5, 200)
        .await
        .unwrap();
    let bot = TelegramStub::new(vec![]);
    reverse_ban_case(&bot.bot, &runtime, case.clone(), HOST_ID, "Host")
        .await
        .unwrap();
    assert!(!runtime
        .queue_network_catchup(&case.id, -300, 5, 200)
        .await
        .unwrap());
    drain(&bot, &runtime).await;
    assert_eq!(calls(&bot, "banchatmember"), 0);
    assert_eq!(notices(&bot), 0);
    assert_eq!(state(&runtime).await.0, "cancelled");
}

#[tokio::test]
async fn policy_and_member_changes_cancel_or_defer_before_banning() {
    for change in ["white", "module", "host", "admin", "lookup_error"] {
        let (runtime, case) = setup().await;
        runtime
            .queue_network_catchup(&case.id, -300, 5, 200)
            .await
            .unwrap();
        match change {
            "white" => runtime
                .set_group_whitelist(-300, 200, true, None)
                .await
                .unwrap(),
            "module" => runtime
                .set_group_module(-300, "netban", false)
                .await
                .unwrap(),
            "host" => runtime.set_maintainer(200, true, None).await.unwrap(),
            _ => {}
        }
        let bot = if change == "lookup_error" {
            TelegramStub::with_failures(vec![], vec![("getchatmember".into(), -300)])
        } else {
            TelegramStub::new(if change == "admin" { vec![200] } else { vec![] })
        };
        drain(&bot, &runtime).await;
        assert_eq!(calls(&bot, "banchatmember"), 0, "{change}");
        assert_eq!(calls(&bot, "sendmessage"), 0);
        if change == "lookup_error" {
            assert_eq!(state(&runtime).await.0, "pending");
            due(&runtime).await;
            drain(&TelegramStub::new(vec![]), &runtime).await;
            assert_eq!(state(&runtime).await.2, Some(100));
        } else {
            assert_eq!(state(&runtime).await.0, "cancelled", "{change}");
        }
    }
}

#[tokio::test]
async fn atomic_enqueue_concurrency_and_rollback() {
    let (runtime, case) = setup().await;
    runtime.with_conn(|c|{c.execute_batch("CREATE TRIGGER refuse_catchup BEFORE INSERT ON network_deliveries BEGIN SELECT RAISE(ABORT,'injected'); END;")?;Ok(())}).await.unwrap();
    assert!(runtime
        .queue_network_catchup(&case.id, -300, 5, 200)
        .await
        .is_err());
    runtime
        .with_conn(|c| {
            assert_eq!(
                c.query_row("SELECT COUNT(*) FROM network_catchups", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            c.execute_batch("DROP TRIGGER refuse_catchup;")?;
            Ok(())
        })
        .await
        .unwrap();
    let (a, b) = tokio::join!(
        runtime.queue_network_catchup(&case.id, -300, 5, 200),
        runtime.queue_network_catchup(&case.id, -300, 5, 200)
    );
    assert!(a.unwrap() && b.unwrap());
    let bot = TelegramStub::new(vec![]);
    let (a, b) = tokio::join!(drain(&bot, &runtime), drain(&bot, &runtime));
    let _ = (a, b);
    assert_eq!(calls(&bot, "banchatmember"), 1);
    assert_eq!(calls(&bot, "sendmessage"), 1);
}

#[tokio::test]
async fn cooldown_and_reversal_after_sent_notice_survive_restart() {
    let (runtime, case) = setup().await;
    runtime
        .queue_network_catchup(&case.id, -300, 5, 200)
        .await
        .unwrap();
    let limited = TelegramStub::with_api_errors(
        vec![],
        vec![(
            "getchatmember".into(),
            -300,
            serde_json::json!({"ok":false,"error_code":429,"description":"Too Many Requests: retry after 120","parameters":{"retry_after":120}}),
        )],
    );
    drain(&limited, &runtime).await;
    assert_eq!(calls(&limited, "banchatmember"), 0);
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due(&restarted).await;
    let good = TelegramStub::new(vec![]);
    drain(&good, &restarted).await;
    assert!(good.requests.lock().unwrap().is_empty());
    restarted
        .with_conn(|c| {
            c.execute("UPDATE telegram_retry_state SET not_before=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
    drain(&good, &restarted).await;
    reverse_ban_case(&good.bot, &restarted, case.clone(), HOST_ID, "Host")
        .await
        .unwrap();
    due(&restarted).await;
    drain(&good, &restarted).await;
    assert_eq!(state(&restarted).await.0, "cancelled");
    assert_eq!(notices(&good), 1);
    assert!(calls(&good, "deletemessage") >= 2);
}

#[tokio::test]
async fn message_and_join_entry_points_save_failed_work_instead_of_claiming_success() {
    for joined in [false, true] {
        let (runtime, case) = setup().await;
        let runtime = Arc::new(runtime);
        let message: Message =
            serde_json::from_value(serde_json::json!({"message_id":5,"date":0,"text":"hello",
            "chat":{"id":-300,"type":"supergroup","title":"Test"},
            "from":{"id":200,"is_bot":false,"first_name":"Test"}}))
            .unwrap();
        let bot = TelegramStub::with_failures(vec![], vec![("banchatmember".into(), -300)]);
        if joined {
            process_new_group_member(&bot.bot, &runtime, &message, message.from.as_ref().unwrap())
                .await;
        } else {
            assert!(check_netban_and_act(&bot.bot, &runtime, &message).await);
        }
        assert_eq!(calls(&bot, "banchatmember"), 1);
        assert_eq!(calls(&bot, "sendmessage"), 0);
        assert_eq!(calls(&bot, "deletemessage"), 0);
        assert!(runtime
            .list_network_ban_targets(&case.id)
            .await
            .unwrap()
            .is_empty());
        assert_eq!(state(&runtime).await.0, "pending");
        let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
        due(&restarted).await;
        let good = TelegramStub::new(vec![]);
        drain(&good, &restarted).await;
        assert_eq!(calls(&good, "banchatmember"), 1);
        assert_eq!(calls(&good, "sendmessage"), 1);
    }
}

#[tokio::test]
async fn upgrade_from_35_preserves_data_and_restores_old_schema() {
    let (runtime, _) = setup().await;
    runtime
        .with_conn(|c| {
            c.execute_batch("DROP TABLE network_catchups; PRAGMA user_version=35;")?;
            Ok(())
        })
        .await
        .unwrap();
    let result = crate::maintenance::check_upgrade(
        &runtime.config.sqlite_path,
        &runtime.config.data_dir.join("catchup-upgrade"),
    )
    .unwrap();
    let result = serde_json::to_value(result).unwrap();
    assert_eq!(result["schema_before"], 35);
    assert_eq!(result["schema_after"], 36);
    assert_eq!(result["restore"], "ok");
}
