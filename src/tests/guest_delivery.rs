use super::*;

fn guest_message() -> Message {
    serde_json::from_value(serde_json::json!({
        "message_id":10,"date":1,"text":"guest advert",
        "chat":{"id":-100,"type":"supergroup","title":"Test"},
        "from":{"id":200,"is_bot":true,"first_name":"Guest","username":"advertbot"}
    }))
    .unwrap()
}

fn telegram(fail: bool) -> TelegramStub {
    let failures = if fail {
        vec![(
            "banchatmember".into(),
            -100,
            serde_json::json!({"ok":false,"error_code":400,"description":"Bad Request: injected failure"}),
        )]
    } else {
        vec![]
    };
    let telegram = TelegramStub::with_members(vec![], failures, true);
    telegram.members.lock().unwrap().insert(
        (-100, 200),
        serde_json::json!({
            "status":"left","user":{"id":200,"is_bot":true,"first_name":"Guest"}
        }),
    );
    telegram
}

async fn runtime(invoker: bool) -> Arc<Runtime> {
    let runtime = Arc::new(test_runtime().await);
    runtime.me_id.set(UserId(999)).unwrap();
    if invoker {
        runtime
            .record_recent_message(-100, 201, MessageId(9), "Invoker", "@advertbot")
            .await;
    }
    runtime
}

async fn cases(runtime: &Runtime) -> Vec<CaseRecord> {
    let ids = runtime
        .with_conn(|conn| {
            let mut stmt = conn.prepare("SELECT id FROM cases ORDER BY target_user_id")?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
        })
        .await
        .unwrap();
    let mut cases = Vec::new();
    for id in ids {
        cases.push(runtime.load_case(&id).await.unwrap().unwrap());
    }
    cases
}

async fn due_now(runtime: &Runtime) {
    runtime.with_conn(|conn| {
        conn.execute_batch("UPDATE origin_ban_jobs SET next_attempt_at=0; UPDATE telegram_retry_state SET not_before=0;")?;
        Ok(())
    }).await.unwrap();
}

fn count(telegram: &TelegramStub, method: &str) -> usize {
    telegram
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter(|(m, _)| m == method)
        .count()
}

#[tokio::test]
async fn failed_guest_and_invoker_bans_are_not_announced_or_propagated_as_success() {
    let runtime = runtime(true).await;
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    let failing = telegram(true);
    assert!(check_guest_bot_and_act(&failing.bot, &runtime, &guest_message()).await);
    let pending = cases(&runtime).await;
    assert_eq!(pending.len(), 2);
    assert!(pending.iter().all(|c| c.status == "ban_failed"));
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_none());
    assert!(runtime
        .find_active_network_ban(201)
        .await
        .unwrap()
        .is_none());
    assert!(!failing
        .requests
        .lock()
        .unwrap()
        .iter()
        .any(|(m, args)| m == "banchatmember" && args["chat_id"] == -300));
    assert!(!failing
        .requests
        .lock()
        .unwrap()
        .iter()
        .any(|(_, args)| args["text"].as_str().unwrap_or("").contains("已封鎖")));
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due_now(&restarted).await;
    let success = telegram(false);
    crate::origin_retry::retry_origin_bans(&success.bot, &restarted)
        .await
        .unwrap();
    let done = cases(&restarted).await;
    assert_eq!(done[0].status, "guest_bot_banned");
    assert_eq!(done[1].status, "guest_invoker_banned");
    assert_eq!(count(&success, "banchatmember"), 2);
    assert_eq!(count(&success, "deletemessage"), 0);
    deliver_network_bans(&success.bot, &restarted, None)
        .await
        .unwrap();
    assert_eq!(count(&success, "banchatmember"), 4);
}

#[tokio::test]
async fn guest_and_invoker_intents_survive_restart_before_either_ban() {
    let runtime = runtime(true).await;
    runtime.delay_telegram_queue(300).await.unwrap();
    let telegram = telegram(false);
    check_guest_bot_and_act(&telegram.bot, &runtime, &guest_message()).await;
    assert_eq!(count(&telegram, "banchatmember"), 0);
    assert_eq!(count(&telegram, "deletemessage"), 0);
    let restarted = Arc::new(Runtime::load(runtime.config.clone()).await.unwrap());
    assert_eq!(cases(&restarted).await.len(), 2);
    due_now(&restarted).await;
    crate::origin_retry::retry_origin_bans(&telegram.bot, &restarted)
        .await
        .unwrap();
    assert_eq!(count(&telegram, "banchatmember"), 2);
    // Re-delivery of the same guest message, including after restart, must
    // neither reopen its case nor send the same notices again.
    restarted.me_id.set(UserId(999)).unwrap();
    let sent = count(&telegram, "sendmessage");
    check_guest_bot_and_act(&telegram.bot, &restarted, &guest_message()).await;
    assert_eq!(cases(&restarted).await.len(), 2);
    assert_eq!(count(&telegram, "banchatmember"), 2);
    assert_eq!(count(&telegram, "sendmessage"), sent);
}

#[tokio::test]
async fn guest_batch_rolls_back_if_the_invoker_job_cannot_be_saved() {
    let runtime = runtime(true).await;
    runtime
        .with_conn(|conn| {
            conn.execute_batch(
                "CREATE TRIGGER fail_invoker BEFORE INSERT ON origin_ban_jobs
            WHEN (SELECT action FROM cases WHERE id=NEW.case_id)='guest_invoker_ban'
            BEGIN SELECT RAISE(ABORT,'injected'); END;",
            )?;
            Ok(())
        })
        .await
        .unwrap();
    let telegram = telegram(false);
    check_guest_bot_and_act(&telegram.bot, &runtime, &guest_message()).await;
    assert!(cases(&runtime).await.is_empty());
    assert_eq!(count(&telegram, "banchatmember"), 0);
    assert_eq!(count(&telegram, "deletemessage"), 0);
    assert!(runtime
        .find_recent_guest_invoker(-100, "advertbot")
        .await
        .is_some());
    runtime
        .with_conn(|conn| {
            conn.execute_batch("DROP TRIGGER fail_invoker;")?;
            Ok(())
        })
        .await
        .unwrap();
    check_guest_bot_and_act(&telegram.bot, &runtime, &guest_message()).await;
    assert_eq!(cases(&runtime).await.len(), 2);
    assert_eq!(count(&telegram, "banchatmember"), 2);
}

#[tokio::test]
async fn retry_skips_a_guest_that_has_since_become_a_member_or_module_was_disabled() {
    for membership_change in [true, false] {
        let runtime = runtime(false).await;
        runtime.delay_telegram_queue(300).await.unwrap();
        let telegram = telegram(false);
        check_guest_bot_and_act(&telegram.bot, &runtime, &guest_message()).await;
        if membership_change {
            telegram.members.lock().unwrap().insert(
                (-100, 200),
                serde_json::json!({
                    "status":"member","user":{"id":200,"is_bot":true,"first_name":"Guest"}
                }),
            );
        } else {
            runtime
                .set_group_module(-100, "guestban", false)
                .await
                .unwrap();
        }
        let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
        due_now(&restarted).await;
        crate::origin_retry::retry_origin_bans(&telegram.bot, &restarted)
            .await
            .unwrap();
        assert_eq!(count(&telegram, "banchatmember"), 0);
        assert_eq!(count(&telegram, "deletemessage"), 0);
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
}

#[tokio::test]
async fn guest_reversal_cancels_saved_work_and_prevents_message_replay() {
    let runtime = runtime(false).await;
    runtime.delay_telegram_queue(300).await.unwrap();
    let telegram = telegram(false);
    check_guest_bot_and_act(&telegram.bot, &runtime, &guest_message()).await;
    let case = cases(&runtime).await.remove(0);
    due_now(&runtime).await;
    reverse_ban_case(&telegram.bot, &runtime, case.clone(), HOST_ID, "Host")
        .await
        .unwrap();
    crate::origin_retry::retry_origin_bans(&telegram.bot, &runtime)
        .await
        .unwrap();
    check_guest_bot_and_act(&telegram.bot, &runtime, &guest_message()).await;
    assert_eq!(count(&telegram, "banchatmember"), 0);
    assert_eq!(
        runtime.load_case(&case.id).await.unwrap().unwrap().status,
        "reversed"
    );
}
