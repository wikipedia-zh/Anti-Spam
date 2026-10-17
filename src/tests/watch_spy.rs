use super::*;
use crate::watch_spy::{process, review_with_bridge, Incoming};

fn incoming(user: i64, text: &str) -> Incoming {
    Incoming {
        id: Uuid::new_v4().to_string(),
        chat: -10003,
        message: 42,
        user,
        text: text.into(),
    }
}

fn bridge() -> Connection {
    let c = Connection::open_in_memory().unwrap();
    c.execute_batch("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);CREATE TABLE sudo_users(user_id INTEGER);
        INSERT INTO meta VALUES ('binding','{\"account_id\":11,\"hex_bot_id\":12}'),('watch_owner','5172206551'),
        ('spy_settings','{\"destination\":-10001,\"excluded\":[-10002],\"bot_id\":13}');
        INSERT INTO sudo_users VALUES (600);").unwrap();
    c.execute(
        "INSERT INTO meta VALUES ('watch_group_verified_at',?1)",
        [Utc::now().timestamp().to_string()],
    )
    .unwrap();
    c
}

async fn model(runtime: &Runtime) {
    let mut m = runtime.model.lock().await;
    m.spam_docs = 100;
    m.ham_docs = 100;
    m.spam_tokens.insert("casino".into(), 1000);
    m.ham_tokens.insert("welcome".into(), 1000);
}

#[tokio::test]
async fn scan_requires_review_and_never_queues_origin_actions() {
    let runtime = test_runtime().await;
    model(&runtime).await;
    runtime.set_threshold(0.85).await.unwrap();
    runtime
        .set_group_module(-10003, "netban", true)
        .await
        .unwrap();
    runtime
        .set_group_module(-10004, "netban", true)
        .await
        .unwrap();
    let item = incoming(200, "casino");
    assert_eq!(
        process(&runtime, &item, -10001, false).await.unwrap().0,
        "review"
    );
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_none());
    let case = runtime.load_case(&item.id).await.unwrap().unwrap();
    assert!(!case.evidence_text.contains("casino"));
    let id = item.id.clone();
    runtime
        .with_conn(move |conn| {
            let b = bridge();
            conn.execute(
                "UPDATE report_deliveries SET review_message_id=10 WHERE case_id=?1",
                [&id],
            )?;
            let tx = conn.transaction()?;
            assert!(review_with_bridge(&tx, &b, &id, "approve", 777, Some((-10001, 10))).is_err());
            assert!(review_with_bridge(&tx, &b, &id, "approve", 600, Some((-10003, 10))).is_err());
            assert!(review_with_bridge(&tx, &b, &id, "approve", 600, Some((-10001, 11))).is_err());
            b.execute("DELETE FROM sudo_users", [])?;
            assert!(review_with_bridge(&tx, &b, &id, "approve", 600, Some((-10001, 10))).is_err());
            b.execute("INSERT INTO sudo_users VALUES (600),(200)", [])?;
            assert!(review_with_bridge(&tx, &b, &id, "approve", 600, Some((-10001, 10))).is_err());
            b.execute("DELETE FROM sudo_users WHERE user_id=200", [])?;
            review_with_bridge(&tx, &b, &id, "approve", 600, Some((-10001, 10)))?;
            assert!(review_with_bridge(&tx, &b, &id, "approve", 600, Some((-10001, 10))).is_err());
            assert_eq!(
                tx.query_row("SELECT count(*) FROM origin_ban_jobs", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            assert_eq!(
                tx.query_row("SELECT count(*) FROM training_samples", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            assert_eq!(
                crate::network_delivery::eligible_target(&tx, &id, -10003, None)?,
                None
            );
            assert_eq!(
                crate::network_delivery::eligible_target(&tx, &id, -10004, None)?,
                Some(200)
            );
            assert_eq!(
                tx.query_row(
                    "SELECT count(*) FROM network_deliveries WHERE case_id=?1 AND chat_id=-10003",
                    [&id],
                    |r| r.get::<_, i64>(0)
                )?,
                0
            );
            crate::reversal_retry::begin_reversal_tx(&tx, &case, HOST_ID, "host")?;
            assert_eq!(
                tx.query_row(
                    "SELECT count(*) FROM network_ban_targets WHERE case_id=?1 AND chat_id=-10003",
                    [&id],
                    |r| r.get::<_, i64>(0)
                )?,
                0
            );
            tx.commit()?;
            Ok(())
        })
        .await
        .unwrap();
    assert_eq!(
        process(&runtime, &item, -10001, false).await.unwrap().0,
        "done"
    );
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn scan_uses_global_threshold_and_exemptions_without_training() {
    let runtime = test_runtime().await;
    model(&runtime).await;
    runtime
        .set_group_threshold(-10003, Some(0.01))
        .await
        .unwrap();
    assert_eq!(
        process(&runtime, &incoming(200, "welcome"), -10001, false)
            .await
            .unwrap()
            .0,
        "ham"
    );
    assert_eq!(
        process(&runtime, &incoming(200, "casino"), -10001, true)
            .await
            .unwrap()
            .0,
        "exempt"
    );
    let item = incoming(200, "casino");
    process(&runtime, &item, -10001, false).await.unwrap();
    assert_eq!(
        process(&runtime, &incoming(200, "casino"), -10001, false)
            .await
            .unwrap()
            .0,
        "exempt"
    );
    runtime
        .with_conn(move |conn| {
            let b = bridge();
            conn.execute(
                "UPDATE report_deliveries SET review_message_id=10 WHERE case_id=?1",
                [&item.id],
            )?;
            let tx = conn.transaction()?;
            review_with_bridge(&tx, &b, &item.id, "reject", HOST_ID, Some((-10001, 10)))?;
            assert_eq!(
                tx.query_row("SELECT count(*) FROM training_samples", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            assert_eq!(
                tx.query_row("SELECT count(*) FROM origin_ban_jobs", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            tx.commit()?;
            Ok(())
        })
        .await
        .unwrap();
}

#[test]
fn report_case_queries_require_trusted_actor_and_pinned_location() {
    let conn = bridge();
    let message = |actor, chat, text: &str| {
        let mut value =
            serde_json::to_value(spam_ban_message(actor, text, Some("/ban 200 +spb"))).unwrap();
        value["chat"]["id"] = serde_json::json!(chat);
        serde_json::from_value::<Message>(value).unwrap()
    };
    let before = conn.total_changes();
    for text in [
        "/case #aab3bf04",
        "/case@WikiBlacklistRobot #aab3bf04",
        "/lookup #aab3bf04",
    ] {
        for actor in [HOST_ID, 600] {
            assert_eq!(
                crate::watch_spy::report_query(&conn, &message(actor, -10001, text))
                    .unwrap()
                    .as_deref(),
                Some("#aab3bf04")
            );
        }
    }
    for (actor, chat, text) in [
        (777, -10001, "/case #aab3bf04"),
        (HOST_ID, -10002, "/case #aab3bf04"),
        (HOST_ID, -10001, "/case@OtherBot #aab3bf04"),
        (HOST_ID, -10001, "/ban 200"),
        (HOST_ID, -10001, "/unban #aab3bf04"),
        (HOST_ID, -10001, "/sync sync:841a90626f07"),
    ] {
        assert!(
            crate::watch_spy::report_query(&conn, &message(actor, chat, text))
                .unwrap()
                .is_none()
        );
    }
    let mut bot = message(HOST_ID, -10001, "/case #aab3bf04");
    bot.from.as_mut().unwrap().is_bot = true;
    assert!(crate::watch_spy::report_query(&conn, &bot)
        .unwrap()
        .is_none());
    let mut forwarded = serde_json::to_value(message(HOST_ID, -10001, "/case #aab3bf04")).unwrap();
    forwarded["forward_origin"] =
        serde_json::json!({"type":"hidden_user", "date":0, "sender_user_name":"Someone"});
    assert!(
        crate::watch_spy::report_query(&conn, &serde_json::from_value(forwarded).unwrap())
            .unwrap()
            .is_none()
    );
    assert_eq!(before, conn.total_changes());
    conn.execute("DELETE FROM sudo_users", []).unwrap();
    assert!(
        crate::watch_spy::report_query(&conn, &message(600, -10001, "/case #aab3bf04"))
            .unwrap()
            .is_none()
    );
    conn.execute(
        "UPDATE meta SET value='0' WHERE key='watch_group_verified_at'",
        [],
    )
    .unwrap();
    assert!(
        crate::watch_spy::report_query(&conn, &message(HOST_ID, -10001, "/case #aab3bf04"))
            .is_err()
    );
}
