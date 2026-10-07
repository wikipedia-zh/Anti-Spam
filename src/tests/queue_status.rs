use super::*;

#[tokio::test]
async fn queue_status_excludes_completed_work_and_tracks_changed_review_notices() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::Mute, -100, 200, Utc::now());
    runtime
        .queue_restriction(case.clone(), 1, None, "mute")
        .await
        .unwrap();
    runtime.with_conn(|conn| {
        conn.execute_batch("INSERT INTO review_updates(case_id,kind,decision,chat_id,message_id,review_status,confirmation_status)
            SELECT id,'train','reject',-1,10,status||':',status||':' FROM cases;")?;
        Ok(())
    }).await.unwrap();
    let initial = runtime.queue_status(None).await.unwrap();
    assert!(initial.contains("未完成：1"));
    assert!(!initial.contains("審核通知"));
    runtime
        .with_conn(|conn| {
            conn.execute_batch(
                "UPDATE restriction_jobs SET state='done'; UPDATE cases SET status='done';",
            )?;
            Ok(())
        })
        .await
        .unwrap();
    let updated = runtime.queue_status(None).await.unwrap();
    assert!(updated.contains("未完成：1"));
    assert!(updated.contains("審核通知"));
    assert!(!updated.contains("禁言／踢人"));
    runtime
        .with_conn(|conn| {
            conn.execute_batch(
                "UPDATE review_updates SET review_status='done:',confirmation_status='done:';",
            )?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(runtime
        .queue_status(None)
        .await
        .unwrap()
        .contains("沒有未完成"));
}

#[tokio::test]
async fn queue_errors_are_bounded_escaped_and_redacted_and_cooldown_is_visible() {
    let runtime = test_runtime().await;
    for message_id in 0..7 {
        runtime
            .queue_restriction(
                dummy_case(ActionKind::Mute, -100, 200, Utc::now()),
                message_id,
                None,
                "mute",
            )
            .await
            .unwrap();
    }
    let secret = runtime.config.bot_token.clone();
    let error = format!("<error> {secret} {}", "x".repeat(3000));
    runtime
        .with_conn(move |conn| {
            conn.execute(
                "UPDATE restriction_jobs SET last_error=?1,attempts=2",
                params![error],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    runtime.delay_telegram_queue(120).await.unwrap();
    let text = runtime.queue_status(None).await.unwrap();
    assert!(text.contains("未完成：7"));
    assert!(text.contains("待執行 0 · 有錯誤 7"));
    assert!(text.contains("Telegram 限流"));
    assert!(text.contains("&lt;error&gt;"));
    assert!(!text.contains(&runtime.config.bot_token));
    assert!(text.contains("[redacted]"));
    assert_eq!(text.matches("錯誤：").count(), 5);
    assert!(text.encode_utf16().count() < 4000);
    assert!(runtime
        .queue_status(Some("no-such-case"))
        .await
        .unwrap()
        .contains("未完成：0"));
}

#[tokio::test]
async fn queue_details_are_only_returned_to_maintainers_in_private_chat() {
    let runtime = Arc::new(test_runtime().await);
    let telegram = TelegramStub::new(vec![555]);
    for (actor, private, allowed) in [
        (555, true, false),
        (HOST_ID, false, false),
        (HOST_ID, true, true),
    ] {
        let mut value = serde_json::to_value(spam_ban_message(actor, "/queue", None)).unwrap();
        if private {
            value["chat"] = serde_json::json!({"id":actor,"type":"private","first_name":"User"});
        }
        telegram.requests.lock().unwrap().clear();
        let ModerationCommand::Queue(id) = parse_command("/queue") else {
            panic!("queue command not recognized");
        };
        crate::queue_status::handle(
            &telegram.bot,
            &runtime,
            &serde_json::from_value(value).unwrap(),
            &id,
        )
        .await
        .unwrap();
        let requests = telegram.requests.lock().unwrap();
        assert_eq!(
            requests.iter().any(|(m, v)| m == "sendmessage"
                && v["text"].as_str().unwrap_or("").contains("工作佇列</b>")),
            allowed
        );
    }
}
