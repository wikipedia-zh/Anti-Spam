use super::*;

fn edited(text: &str) -> Message {
    serde_json::from_value(serde_json::json!({
        "message_id":10,"date":1,"edit_date":2,
        "chat":{"id":-100,"type":"supergroup","title":"Test"},
        "from":{"id":200,"is_bot":false,"first_name":"Member"},
        "text":text
    }))
    .unwrap()
}

async fn runtime() -> Arc<Runtime> {
    let runtime = Arc::new(test_runtime().await);
    runtime.me_id.set(UserId(999)).unwrap();
    runtime.add_spam_rule("buy-now", "Test rule").await.unwrap();
    runtime
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
async fn harmless_text_edited_into_spam_is_removed() {
    let runtime = runtime().await;
    let telegram = TelegramStub::new(vec![999]);
    moderate_edited_message(
        telegram.bot.clone(),
        runtime.clone(),
        edited("Hello everyone"),
    )
    .await
    .unwrap();
    assert!(calls(&telegram, "banchatmember").is_empty());
    moderate_edited_message(telegram.bot.clone(), runtime.clone(), edited("buy-now"))
        .await
        .unwrap();
    assert_eq!(calls(&telegram, "banchatmember").len(), 1);
    assert!(calls(&telegram, "deletemessage")
        .iter()
        .any(|r| r["message_id"] == 10));
    let case = runtime
        .find_active_ban_in_chat(-100, 200)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(case.source_message_id, Some(10));
    assert!(case.evidence_text.contains("buy-now"));
}

#[tokio::test]
async fn changed_captions_and_hidden_links_are_checked() {
    for hidden_link in [false, true] {
        let runtime = runtime().await;
        let telegram = TelegramStub::new(vec![999]);
        let mut value = serde_json::to_value(edited("safe label")).unwrap();
        if hidden_link {
            value["entities"] = serde_json::json!([{"type":"text_link","offset":0,"length":10,"url":"https://buy-now.example/"}]);
        } else {
            value.as_object_mut().unwrap().remove("text");
            value["document"] =
                serde_json::json!({"file_id":"f","file_unique_id":"u","file_name":"notes.pdf"});
            value["caption"] = serde_json::json!("buy-now");
        }
        moderate_edited_message(
            telegram.bot.clone(),
            runtime.clone(),
            serde_json::from_value(value).unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(
            calls(&telegram, "banchatmember").len(),
            1,
            "hidden_link={hidden_link}"
        );
    }
}

#[tokio::test]
async fn editing_commands_does_not_execute_them_or_increment_flood_counts() {
    let runtime = runtime().await;
    let telegram = TelegramStub::new(vec![999, HOST_ID]);
    let mut message = edited("/module all off");
    message.from.as_mut().unwrap().id = UserId(HOST_ID as u64);
    for _ in 0..6 {
        moderate_edited_message(telegram.bot.clone(), runtime.clone(), message.clone())
            .await
            .unwrap();
    }
    let settings = runtime.get_group_modules(-100).await.unwrap();
    assert!(settings.flood_control);
    assert!(runtime.flood_tracker.lock().await.is_empty());
    assert!(calls(&telegram, "restrictchatmember").is_empty());
    assert!(calls(&telegram, "sendmessage").is_empty());
    assert_eq!(
        runtime
            .with_conn(|conn| Ok(conn.query_row(
                "SELECT COUNT(*) FROM maintainer_actions",
                [],
                |r| r.get::<_, i64>(0)
            )?))
            .await
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn edits_keep_test_groups_score_only_and_honor_exemptions() {
    for exemption in ["test_group", "admin", "whitelist", "private", "terminated"] {
        let mut loaded = test_runtime().await;
        loaded.me_id.set(UserId(999)).unwrap();
        if exemption == "test_group" {
            loaded.config.test_group_id = Some(-100);
        }
        let runtime = Arc::new(loaded);
        runtime.add_spam_rule("buy-now", "Test rule").await.unwrap();
        let telegram = TelegramStub::new(if exemption == "admin" {
            vec![999, 200]
        } else {
            vec![999]
        });
        let mut message = edited("buy-now");
        match exemption {
            "whitelist" => {
                runtime.with_conn(|conn| { conn.execute("INSERT INTO group_whitelist(chat_id,user_id,created_at) VALUES (-100,200,'now')",[])?; Ok(()) }).await.unwrap();
            }
            "private" => {
                let mut value = serde_json::to_value(message).unwrap();
                value["chat"] = serde_json::json!({"id":200,"type":"private","first_name":"Test"});
                message = serde_json::from_value(value).unwrap();
            }
            "terminated" => {
                runtime.banned_groups.write().await.insert(-100);
            }
            _ => {}
        }
        moderate_edited_message(telegram.bot.clone(), runtime.clone(), message)
            .await
            .unwrap();
        assert!(calls(&telegram, "banchatmember").is_empty(), "{exemption}");
        assert!(calls(&telegram, "deletemessage").is_empty(), "{exemption}");
    }
}

#[tokio::test]
async fn attachment_failure_does_not_create_a_successful_ban() {
    let runtime = runtime().await;
    runtime
        .set_group_module(-100, "noexec", true)
        .await
        .unwrap();
    let telegram = TelegramStub::with_failures(vec![999], vec![("banchatmember".into(), -100)]);
    let mut value = serde_json::to_value(edited("")).unwrap();
    value.as_object_mut().unwrap().remove("text");
    value["document"] =
        serde_json::json!({"file_id":"f","file_unique_id":"u","file_name":"invoice.exe"});
    moderate_edited_message(
        telegram.bot.clone(),
        runtime.clone(),
        serde_json::from_value(value).unwrap(),
    )
    .await
    .unwrap();
    assert!(runtime
        .find_active_ban_in_chat(-100, 200)
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        runtime
            .with_conn(|conn| Ok(conn.query_row(
                "SELECT COUNT(*) FROM cases WHERE status='ban_failed'",
                [],
                |r| r.get::<_, i64>(0)
            )?))
            .await
            .unwrap(),
        1
    );
    assert_eq!(calls(&telegram, "banchatmember").len(), 1);
}

#[tokio::test]
async fn join_guard_failure_does_not_publish_a_successful_ban() {
    let runtime = runtime().await;
    runtime
        .set_group_module(-100, "nohalal", true)
        .await
        .unwrap();
    let telegram = TelegramStub::with_failures(vec![999], vec![("banchatmember".into(), -100)]);
    let mut message = edited("");
    message.from.as_mut().unwrap().first_name = "حسن".to_string();
    process_new_group_member(
        &telegram.bot,
        &runtime,
        &message,
        message.from.as_ref().unwrap(),
    )
    .await;
    assert!(runtime
        .find_active_ban_in_chat(-100, 200)
        .await
        .unwrap()
        .is_none());
    assert_eq!(calls(&telegram, "banchatmember").len(), 1);
    assert_eq!(
        runtime
            .with_conn(|conn| Ok(conn.query_row(
                "SELECT COUNT(*) FROM cases WHERE status='ban_failed'",
                [],
                |r| r.get::<_, i64>(0)
            )?))
            .await
            .unwrap(),
        1
    );
}
