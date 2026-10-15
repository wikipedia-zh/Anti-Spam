use super::*;
use crate::case_ids as ids;
use serde_json::json;

#[tokio::test]
async fn panel_and_queue_keep_full_keys_while_accepting_short_searches() {
    let runtime = test_runtime().await;
    let mut case = dummy_case(ActionKind::Mute, -100, 200, Utc::now());
    case.id = "12345678-1111-4111-8111-111111111111".into();
    runtime
        .queue_restriction(case.clone(), 1, None, "mute")
        .await
        .unwrap();
    let short = ids::display(&case.id);
    for view in ["cases", "queue"] {
        let result = runtime
            .host_query(serde_json::from_value(json!({"view":view,"search":short})).unwrap())
            .await
            .unwrap();
        let items = result["items"].as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["short_case_id"], short);
        assert_eq!(
            items[0][if view == "cases" { "id" } else { "case_id" }],
            case.id
        );
    }
    let detail = runtime
        .host_case(crate::host_cases::Query {
            case_id: short.clone(),
            offset: 0,
        })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(detail["case"]["id"], case.id);
    assert_eq!(detail["case"]["short_id"], short);
    assert_eq!(
        runtime.queue_status(Some(&short)).await.unwrap(),
        runtime.queue_status(Some(&case.id)).await.unwrap()
    );
    let mut other = case.clone();
    other.id = "12345678-2222-4222-8222-222222222222".into();
    runtime.persist_case(&other).await.unwrap();
    let result = runtime
        .host_query(serde_json::from_value(json!({"view":"cases","search":short})).unwrap())
        .await
        .unwrap();
    assert_eq!(result["items"].as_array().unwrap().len(), 2);
    assert!(runtime
        .load_case_reference(&short)
        .await
        .unwrap_err()
        .downcast_ref::<ids::Ambiguous>()
        .is_some());
    assert!(runtime.queue_status(Some(&short)).await.is_err());
    assert_eq!(
        runtime.load_case(&case.id).await.unwrap().unwrap().id,
        case.id
    );
}

#[test]
fn short_unban_rejects_collisions_and_numeric_codes_do_not_target_a_user_id() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async {
                    let runtime = Arc::new(test_runtime().await);
                    let mut case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
                    case.id = "12345678-1111-4111-8111-111111111111".into();
                    runtime.persist_case(&case).await.unwrap();
                    let mut other = case.clone();
                    other.id = "12345678-2222-4222-8222-222222222222".into();
                    other.target_user_id = 201;
                    runtime.persist_case(&other).await.unwrap();
                    let telegram = TelegramStub::new(vec![]);
                    handle_command(
                        telegram.bot.clone(),
                        runtime.clone(),
                        spam_ban_message(HOST_ID, "/unban #12345678", None),
                    )
                    .await
                    .unwrap();
                    let calls = telegram.requests.lock().unwrap().clone();
                    assert!(!calls.iter().any(|(method, _)| method == "unbanchatmember"));
                    assert!(calls
                        .iter()
                        .any(|(_, args)| args["text"].as_str() == Some(ids::AMBIGUOUS)));
                    assert_ne!(
                        runtime.load_case(&case.id).await.unwrap().unwrap().status,
                        "reversed"
                    );
                    // A longer reference selects one case; internal reversal still uses its full UUID.
                    handle_command(
                        telegram.bot.clone(),
                        runtime.clone(),
                        spam_ban_message(HOST_ID, "/unban #123456781", None),
                    )
                    .await
                    .unwrap();
                    let calls = telegram.requests.lock().unwrap().clone();
                    let unbans: Vec<_> = calls
                        .iter()
                        .filter(|(method, _)| method == "unbanchatmember")
                        .collect();
                    assert!(!unbans.is_empty());
                    assert!(unbans.iter().all(|(_, args)| args["user_id"] == 200));
                    assert_eq!(
                        runtime.load_case(&case.id).await.unwrap().unwrap().status,
                        "reversed"
                    );
                    assert_ne!(
                        runtime.load_case(&other.id).await.unwrap().unwrap().status,
                        "reversed"
                    );
                    // Full IDs in old messages still work after a short-code reversal.
                    handle_command(
                        telegram.bot.clone(),
                        runtime.clone(),
                        spam_ban_message(HOST_ID, &format!("/case {}", case.id), None),
                    )
                    .await
                    .unwrap();
                    assert!(telegram
                        .requests
                        .lock()
                        .unwrap()
                        .iter()
                        .any(|(_, args)| args["text"]
                            .as_str()
                            .is_some_and(|text| text.contains("#12345678"))));
                });
        })
        .unwrap()
        .join()
        .unwrap();
}
