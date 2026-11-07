use super::*;
use crate::seele::{admit, NetbanRequest};

#[tokio::test]
async fn private_project_ban_reverses_groups_without_unbanning_a_dm() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::ProjectBan, HOST_ID, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    runtime.record_network_ban_target(&case.id, -300).await.unwrap();
    runtime.record_network_ban_target(&case.id, HOST_ID).await.unwrap();
    let stub = TelegramStub::new(vec![]);
    reverse_ban_case(&stub.bot, &runtime, case.clone(), HOST_ID, "owner").await.unwrap();
    {
        let calls = stub.requests.lock().unwrap();
        let unbans: Vec<_> = calls.iter().filter(|(m, _)| m == "unbanchatmember").collect();
        assert_eq!(unbans.len(), 1);
        assert_eq!(unbans[0].1["chat_id"], -300);
    }
    assert_eq!(runtime.load_case(&case.id).await.unwrap().unwrap().status, "reversed");
}

fn request(id: String) -> NetbanRequest {
    NetbanRequest {
        request_id: id,
        id: 200,
        operator_id: HOST_ID,
        host_chat_id: -100123,
        host_message_id: 50,
        source_message_id: 10,
    }
}

#[tokio::test]
async fn seele_admission_is_atomic_idempotent_and_reversible() {
    let runtime = test_runtime().await;
    runtime
        .set_group_module(-300, "netban", true)
        .await
        .unwrap();
    let id = Uuid::new_v4().to_string();
    let key = id.clone();
    runtime
        .with_conn(move |conn| {
            let r = request(key.clone());
            assert_eq!(admit(conn, &r, -100123, None)?, (key.clone(), true));
            assert_eq!(admit(conn, &r, -100123, None)?, (key.clone(), false));
            assert_eq!(
                conn.query_row(
                    "SELECT count(*) FROM network_deliveries WHERE case_id=?1",
                    [&key],
                    |r| r.get::<_, i64>(0)
                )?,
                1
            );
            assert_eq!(
                conn.query_row(
                    "SELECT count(*) FROM origin_ban_jobs WHERE case_id=?1",
                    [&key],
                    |r| r.get::<_, i64>(0)
                )?,
                0
            );
            Ok(())
        })
        .await
        .unwrap();
    let case = runtime.load_case(&id).await.unwrap().unwrap();
    assert!(crate::seele::is_case(&case));
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_some());
    let stub = TelegramStub::new(vec![]);
    reverse_ban_case(&stub.bot, &runtime, case, HOST_ID, "owner")
        .await
        .unwrap();
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_none());
    assert!(!stub
        .requests
        .lock()
        .unwrap()
        .iter()
        .any(|(method, args)| method == "unbanchatmember" && args["chat_id"] == 200));
    runtime
        .with_conn(move |conn| {
            assert!(admit(conn, &request(id), -100123, None).is_err());
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn seele_rejects_wrong_host_unauthorized_and_protected_targets() {
    let runtime = test_runtime().await;
    runtime
        .with_conn(|conn| {
            let mut r = request(Uuid::new_v4().to_string());
            assert!(admit(conn, &r, -100999, None).is_err());
            r.operator_id = 321;
            assert!(admit(conn, &r, -100123, None).is_err());
            r.operator_id = HOST_ID;
            r.id = HOST_ID;
            assert!(admit(conn, &r, -100123, None).is_err());
            r.id = 200;
            conn.execute(
                "INSERT INTO global_whitelist(user_id,created_at) VALUES(200,'now')",
                [],
            )?;
            assert!(admit(conn, &r, -100123, None).is_err());
            assert_eq!(
                conn.query_row("SELECT count(*) FROM cases", [], |r| r.get::<_, i64>(0))?,
                0
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn missing_ban_record_is_not_an_unban_confirmation() {
    let runtime = test_runtime().await;
    let stub = TelegramStub::new(vec![]);
    handle_exchange_request_unban(
        &stub.bot,
        &runtime,
        -900,
        serde_json::json!({
            "id":200,"operator_id":HOST_ID,"request_id":"unban:now:200:owner"
        }),
    )
    .await;
    let requests = stub.requests.lock().unwrap();
    let (_, args) = requests.iter().find(|(m, _)| m == "sendmessage").unwrap();
    let reply: serde_json::Value = serde_json::from_str(args["text"].as_str().unwrap()).unwrap();
    assert_eq!(reply["data"]["success"], false);
}

#[tokio::test]
async fn unauthorized_appeal_does_not_unban() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    let stub = TelegramStub::new(vec![]);
    handle_exchange_request_unban(
        &stub.bot,
        &runtime,
        -900,
        serde_json::json!({
            "id":200,"operator_id":999,"request_id":"unban:now:200:999"
        }),
    )
    .await;
    assert!(!stub
        .requests
        .lock()
        .unwrap()
        .iter()
        .any(|(m, _)| m == "unbanchatmember"));
    assert_ne!(
        runtime.load_case(&case.id).await.unwrap().unwrap().status,
        "reversed"
    );
}

#[tokio::test]
async fn appeal_status_database_error_is_not_a_clear_record() {
    let runtime = test_runtime().await;
    runtime
        .with_conn(|conn| {
            conn.execute("DROP TABLE cases", [])?;
            Ok(())
        })
        .await
        .unwrap();
    let stub = TelegramStub::new(vec![]);
    handle_exchange_query_bad(
        &stub.bot,
        &runtime,
        -900,
        serde_json::json!({"id":200,"request_id":"q"}),
    )
    .await;
    handle_exchange_query_bad_detail(
        &stub.bot,
        &runtime,
        -900,
        serde_json::json!({"id":200,"request_id":"q"}),
    )
    .await;
    assert!(stub.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn delayed_unban_confirmation_requires_every_case_and_no_remaining_ban() {
    let runtime = test_runtime().await;
    let mut case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    let data =
        serde_json::json!({"id":200,"case_ids":[case.id],"request_id":"unban:now:200:owner"});
    let stub = TelegramStub::new(vec![]);
    crate::seele::check_unban(&stub.bot, &runtime, -900, data.clone()).await;
    assert!(stub.requests.lock().unwrap().is_empty());
    case.action = ActionKind::Unbanned;
    case.status = "reversed".into();
    runtime.persist_case(&case).await.unwrap();
    crate::seele::check_unban(&stub.bot, &runtime, -900, data.clone()).await;
    assert_eq!(stub.requests.lock().unwrap().len(), 1);
    stub.requests.lock().unwrap().clear();
    let other = dummy_case(ActionKind::SpamBan, -200, 200, Utc::now());
    runtime.persist_case(&other).await.unwrap();
    crate::seele::check_unban(&stub.bot, &runtime, -900, data).await;
    assert!(stub.requests.lock().unwrap().is_empty());
}
