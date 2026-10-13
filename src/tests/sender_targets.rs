use super::*;

const CHANNEL: i64 = -1001234567890;

fn channel_message(command: &str, actor: i64) -> Message {
    serde_json::from_value(serde_json::json!({
        "message_id": 2, "date": 1, "text": command,
        "chat": {"id": -100, "type": "supergroup", "title": "Group"},
        "from": {"id": actor, "is_bot": false, "first_name": "Reporter"},
        "reply_to_message": {
            "message_id": 1, "date": 1, "text": "casino gambling",
            "chat": {"id": -100, "type": "supergroup", "title": "Group"},
            "from": {"id": 136817688, "is_bot": true, "first_name": "Channel", "username": "Channel_Bot"},
            "sender_chat": {"id": CHANNEL, "type": "channel", "title": "Channel <name>", "username": "spam_channel"}
        }
    })).unwrap()
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

async fn only_case(runtime: &Runtime) -> CaseRecord {
    let id = runtime
        .with_conn(|c| Ok(c.query_row("SELECT id FROM cases", [], |r| r.get::<_, String>(0))?))
        .await
        .unwrap();
    runtime.load_case(&id).await.unwrap().unwrap()
}

async fn approve(runtime: &Runtime, case: &CaseRecord) {
    let guard = runtime.review_guard(&case.id).await;
    assert!(runtime
        .decide_report(case, "approve", (HOST_ID, "Host".into()), (-1, 100), guard)
        .await
        .unwrap());
}

async fn due(runtime: &Runtime) {
    runtime.with_conn(|c| {
        c.execute_batch("UPDATE origin_ban_jobs SET next_attempt_at=0; UPDATE network_deliveries SET next_attempt_at=0; UPDATE reversal_retries SET next_attempt_at=0;")?;
        Ok(())
    }).await.unwrap();
}

#[test]
fn channel_report_keeps_identity_through_restart_approval_network_ban_and_unban() {
    // Match the existing command tests: the full dispatcher future is large
    // in debug builds, so construct it on a dedicated test thread.
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async {
                    let runtime = Arc::new(test_runtime().await);
                    runtime
                        .set_group_module(-300, "netban", true)
                        .await
                        .unwrap();
                    let telegram = TelegramStub::new(vec![]);
                    handle_command(
                        telegram.bot.clone(),
                        runtime.clone(),
                        channel_message("/spam", 444),
                    )
                    .await
                    .unwrap();
                    let case = only_case(&runtime).await;
                    assert_eq!(case.target_user_id, CHANNEL);
                    assert_eq!(case.target_name, "Channel <name> (@spam_channel)");
                    assert_eq!(case.source_message_id, Some(1));
                    assert_eq!(case.status, "pending_review");
                    let card = calls(&telegram, "sendmessage")
                        .into_iter()
                        .find(|a| a["chat_id"] == -1)
                        .unwrap();
                    assert!(card["text"]
                        .as_str()
                        .unwrap()
                        .contains("Channel &lt;name&gt; (@spam_channel)"));
                    assert!(!card["text"].as_str().unwrap().contains("Channel_Bot"));
                    assert!(calls(&telegram, "banchatsenderchat").is_empty());

                    approve(&runtime, &case).await;
                    let restarted = Arc::new(Runtime::load(runtime.config.clone()).await.unwrap());
                    crate::origin_retry::retry_origin_bans(&telegram.bot, &restarted)
                        .await
                        .unwrap();
                    deliver_network_bans(&telegram.bot, &restarted, None)
                        .await
                        .unwrap();
                    let saved = restarted.load_case(&case.id).await.unwrap().unwrap();
                    assert_eq!(saved.status, "approved_and_banned");
                    assert_eq!(saved.target_user_id, CHANNEL);
                    assert_eq!(restarted.model.lock().await.spam_docs, 1);
                    let bans = calls(&telegram, "banchatsenderchat");
                    assert_eq!(bans.len(), 2);
                    assert!(bans
                        .iter()
                        .all(|a| a["sender_chat_id"] == CHANNEL && a.get("user_id").is_none()));
                    assert!(bans.iter().any(|a| a["chat_id"] == -100));
                    assert!(bans.iter().any(|a| a["chat_id"] == -300));
                    assert!(calls(&telegram, "getchatmember")
                        .iter()
                        .all(|a| a["user_id"]
                            .as_i64()
                            .is_some_and(|id| id > 0 && id != 136817688)));
                    assert!(calls(&telegram, "banchatmember").is_empty());

                    // Reply-based unban must resolve the same channel instead of Channel_Bot.
                    handle_command(
                        telegram.bot.clone(),
                        restarted.clone(),
                        channel_message("/unban", HOST_ID),
                    )
                    .await
                    .unwrap();
                    assert_eq!(
                        restarted.load_case(&case.id).await.unwrap().unwrap().status,
                        "reversed"
                    );
                    assert_eq!(restarted.model.lock().await.spam_docs, 0);
                    let unbans = calls(&telegram, "unbanchatsenderchat");
                    assert_eq!(unbans.len(), 2);
                    assert!(unbans.iter().all(|a| a["sender_chat_id"] == CHANNEL));
                    assert!(calls(&telegram, "unbanchatmember").is_empty());
                });
        })
        .unwrap()
        .join()
        .unwrap();
}

#[tokio::test]
async fn channel_ban_and_reversal_failures_retry_without_losing_the_signed_target() {
    let runtime = test_runtime().await;
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    let failed = TelegramStub::with_failures(vec![], vec![("banchatsenderchat".into(), -100)]);
    crate::report_delivery::handle(&failed.bot, &runtime, &channel_message("/spam", 444))
        .await
        .unwrap();
    let case = only_case(&runtime).await;
    approve(&runtime, &case).await;
    crate::origin_retry::retry_origin_bans(&failed.bot, &runtime)
        .await
        .unwrap();
    assert_eq!(
        runtime.load_case(&case.id).await.unwrap().unwrap().status,
        "ban_failed"
    );
    assert_eq!(runtime.model.lock().await.spam_docs, 0);
    assert!(calls(&failed, "deletemessage")
        .iter()
        .all(|a| a["message_id"] != 1));

    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due(&restarted).await;
    let partial = TelegramStub::with_failures(vec![], vec![("banchatsenderchat".into(), -300)]);
    crate::origin_retry::retry_origin_bans(&partial.bot, &restarted)
        .await
        .unwrap();
    deliver_network_bans(&partial.bot, &restarted, None)
        .await
        .unwrap();
    assert_eq!(restarted.model.lock().await.spam_docs, 1);
    let success = TelegramStub::new(vec![]);
    due(&restarted).await;
    deliver_network_bans(&success.bot, &restarted, None)
        .await
        .unwrap();
    assert_eq!(
        calls(&success, "banchatsenderchat"),
        vec![serde_json::json!({"chat_id": -300, "sender_chat_id": CHANNEL})]
    );
    let saved = restarted.load_case(&case.id).await.unwrap().unwrap();
    let failed_unban =
        TelegramStub::with_failures(vec![], vec![("unbanchatsenderchat".into(), -300)]);
    assert!(
        reverse_ban_case(&failed_unban.bot, &restarted, saved, HOST_ID, "Host")
            .await
            .is_err()
    );
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due(&restarted).await;
    crate::reversal_retry::retry_due_reversals(&success.bot, &restarted)
        .await
        .unwrap();
    assert_eq!(
        calls(&success, "unbanchatsenderchat"),
        vec![serde_json::json!({"chat_id": -300, "sender_chat_id": CHANNEL})]
    );
    assert_eq!(
        restarted.load_case(&case.id).await.unwrap().unwrap().status,
        "reversed"
    );
}

#[test]
fn sender_chat_wins_over_fake_or_missing_from_but_forward_origins_do_not() {
    let message = channel_message("/spam", 444);
    assert_eq!(
        crate::sender_targets::reply_context(&message).unwrap().0,
        CHANNEL
    );
    let mut no_from = serde_json::to_value(&message).unwrap();
    no_from["reply_to_message"]
        .as_object_mut()
        .unwrap()
        .remove("from");
    assert_eq!(
        crate::sender_targets::reply_context(&serde_json::from_value(no_from).unwrap())
            .unwrap()
            .0,
        CHANNEL
    );

    let mut forwarded =
        serde_json::to_value(spam_ban_message(444, "/spam", Some("forwarded text"))).unwrap();
    forwarded["reply_to_message"]["forward_origin"] = serde_json::json!({"type":"channel", "date":1, "chat":{"id":CHANNEL,"type":"channel","title":"Origin"},"message_id":9});
    let context =
        crate::sender_targets::reply_context(&serde_json::from_value(forwarded).unwrap()).unwrap();
    assert_eq!(context.0, 200);
    assert!(context.3.contains(&format!("[fwd_id: {CHANNEL}]")));
}

#[tokio::test]
async fn anonymous_admins_linked_autoforwards_and_unidentified_pseudo_senders_do_not_become_reports(
) {
    let runtime = test_runtime().await;
    let telegram = TelegramStub::new(vec![]);
    for kind in ["anonymous", "automatic", "missing"] {
        let mut value = serde_json::to_value(channel_message("/spam", 444)).unwrap();
        match kind {
            "anonymous" => {
                value["reply_to_message"]["sender_chat"] =
                    serde_json::json!({"id":-100,"type":"supergroup","title":"Group"})
            }
            "automatic" => {
                value["reply_to_message"]["is_automatic_forward"] = serde_json::json!(true)
            }
            _ => {
                value["reply_to_message"]
                    .as_object_mut()
                    .unwrap()
                    .remove("sender_chat");
            }
        }
        crate::report_delivery::handle(
            &telegram.bot,
            &runtime,
            &serde_json::from_value(value).unwrap(),
        )
        .await
        .unwrap();
    }
    runtime
        .with_conn(|c| {
            assert_eq!(
                c.query_row("SELECT COUNT(*) FROM cases", [], |r| r.get::<_, i64>(0))?,
                0
            );
            Ok(())
        })
        .await
        .unwrap();
    assert!(calls(&telegram, "banchatmember").is_empty());
    assert!(calls(&telegram, "banchatsenderchat").is_empty());
}

#[tokio::test]
async fn linked_channels_and_failed_lookups_are_protected_during_delivery() {
    for lookup_failed in [false, true] {
        let runtime = test_runtime().await;
        let response = if lookup_failed {
            serde_json::json!({"ok":false,"error_code":400,"description":"Bad Request: lookup failed"})
        } else {
            serde_json::json!({"ok":true,"result":{"id":-100,"type":"supergroup","title":"Group","linked_chat_id":CHANNEL,"accent_color_id":0,"max_reaction_count":11,"accepted_gift_types":{"unlimited_gifts":false,"limited_gifts":false,"unique_gifts":false,"premium_subscription":false}}})
        };
        let telegram =
            TelegramStub::with_api_errors(vec![], vec![("getchat".into(), -100, response)]);
        crate::report_delivery::handle(&telegram.bot, &runtime, &channel_message("/spam", 444))
            .await
            .unwrap();
        let case = only_case(&runtime).await;
        approve(&runtime, &case).await;
        crate::origin_retry::retry_origin_bans(&telegram.bot, &runtime)
            .await
            .unwrap();
        assert!(calls(&telegram, "banchatsenderchat").is_empty());
        assert_eq!(runtime.model.lock().await.spam_docs, 0);
    }
}

#[test]
fn local_admin_can_unban_a_channel_without_a_case() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async {
                    let runtime = Arc::new(test_runtime().await);
                    let telegram = TelegramStub::new(vec![444]);
                    handle_command(
                        telegram.bot.clone(),
                        runtime,
                        channel_message("/unban", 444),
                    )
                    .await
                    .unwrap();
                    assert_eq!(
                        calls(&telegram, "unbanchatsenderchat"),
                        vec![serde_json::json!({"chat_id":-100,"sender_chat_id":CHANNEL})]
                    );
                    assert!(calls(&telegram, "unbanchatmember").is_empty());
                });
        })
        .unwrap()
        .join()
        .unwrap();
}
