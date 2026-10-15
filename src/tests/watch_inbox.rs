use super::*;
use crate::watch_inbox::{process, Incoming};

fn message(user: i64, text: &str) -> Incoming {
    Incoming {
        id: Uuid::new_v4().to_string(),
        user,
        text: text.into(),
    }
}

async fn model(runtime: &Runtime) {
    let mut model = runtime.model.lock().await;
    model.spam_docs = 100;
    model.ham_docs = 100;
    model.spam_tokens.insert("casino".into(), 1000);
    model.ham_tokens.insert("welcome".into(), 1000);
}

#[tokio::test]
async fn private_ban_uses_global_threshold_keeps_evidence_private_and_reverses() {
    let runtime = test_runtime().await;
    model(&runtime).await;
    runtime.set_threshold(0.99).await.unwrap();
    runtime
        .set_group_module(-100, "netban", true)
        .await
        .unwrap();
    let incoming = message(200, "casino");
    let result = process(&runtime, &incoming, false).await.unwrap();
    assert_eq!(result.0, "banned");
    let case = runtime.load_case(&incoming.id).await.unwrap().unwrap();
    assert!(!case.evidence_text.contains("casino"));
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_some());
    process(&runtime, &incoming, false).await.unwrap();
    let case_id = incoming.id.clone();
    runtime
        .with_conn(move |conn| {
            assert_eq!(
                conn.query_row(
                    "SELECT count(*) FROM network_deliveries WHERE case_id=?1",
                    [&case_id],
                    |r| r.get::<_, i64>(0)
                )?,
                1
            );
            assert_eq!(
                conn.query_row(
                    "SELECT count(*) FROM origin_ban_jobs WHERE case_id=?1",
                    [&case_id],
                    |r| r.get::<_, i64>(0)
                )?,
                0
            );
            let tx = conn.transaction()?;
            crate::reversal_retry::begin_reversal_tx(&tx, &case, HOST_ID, "host")?;
            assert_eq!(
                tx.query_row(
                    "SELECT count(*) FROM network_ban_targets WHERE chat_id=200",
                    [],
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
        process(&runtime, &incoming, false).await.unwrap().0,
        "exempt"
    );
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn private_ham_and_protected_users_are_not_banned() {
    let runtime = test_runtime().await;
    model(&runtime).await;
    assert_eq!(
        process(&runtime, &message(200, "welcome"), false)
            .await
            .unwrap()
            .0,
        "ham"
    );
    assert_eq!(
        process(&runtime, &message(200, "casino"), true)
            .await
            .unwrap()
            .0,
        "exempt"
    );
    assert_eq!(
        process(&runtime, &message(HOST_ID, "casino"), false)
            .await
            .unwrap()
            .0,
        "exempt"
    );
    runtime
        .with_conn(|conn| {
            conn.execute(
                "INSERT INTO global_whitelist(user_id,created_at) VALUES (200,'2026-10-15')",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    assert_eq!(
        process(&runtime, &message(200, "casino"), false)
            .await
            .unwrap()
            .0,
        "exempt"
    );
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn private_regex_does_not_bypass_model_threshold_and_rejection_does_not_train() {
    let runtime = test_runtime().await;
    model(&runtime).await;
    runtime.set_group_threshold(200, Some(0.01)).await.unwrap();
    runtime.spam_rules.write().await.push(SpamRule {
        id: 1,
        description: "test".into(),
        regex: FancyRegex::new("welcome").unwrap(),
    });
    let incoming = message(200, "welcome");
    assert_eq!(
        process(&runtime, &incoming, false).await.unwrap().0,
        "review"
    );
    let id = incoming.id;
    runtime
        .with_conn(move |conn| {
            let tx = conn.transaction()?;
            assert!(
                crate::review_decisions::report(&tx, &id, "approve", (555, "outsider"), None)
                    .is_err()
            );
            assert!(crate::review_decisions::report(
                &tx,
                &id,
                "reject",
                (HOST_ID, "host"),
                None
            )?);
            assert_eq!(
                tx.query_row(
                    "SELECT count(*) FROM training_samples WHERE case_id=?1",
                    [&id],
                    |r| r.get::<_, i64>(0)
                )?,
                0
            );
            assert_eq!(
                tx.query_row(
                    "SELECT count(*) FROM origin_ban_jobs WHERE case_id=?1",
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
    assert!(runtime
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn private_review_rechecks_sudo_and_only_queues_group_deliveries() {
    let runtime = test_runtime().await;
    runtime
        .set_group_module(-100, "netban", true)
        .await
        .unwrap();
    let incoming = message(200, "unknown");
    assert_eq!(
        process(&runtime, &incoming, false).await.unwrap().0,
        "review"
    );
    let id = incoming.id;
    runtime.with_conn(move |conn| {
        let bridge=Connection::open_in_memory()?;
        bridge.execute_batch("CREATE TABLE meta(key TEXT,value TEXT); CREATE TABLE sudo_users(user_id INTEGER);
            INSERT INTO meta VALUES ('binding','{\"account_id\":11,\"hex_bot_id\":12}'),('watch_owner','5172206551');
            INSERT INTO sudo_users VALUES (200);")?;
        let tx=conn.transaction()?;
        assert!(crate::watch_inbox::approve(&tx,&bridge,&id,200,HOST_ID).is_err());
        bridge.execute("DELETE FROM sudo_users",[])?;
        assert!(crate::watch_inbox::approve(&tx,&bridge,&id,200,555).is_err());
        crate::watch_inbox::approve(&tx,&bridge,&id,200,HOST_ID)?;
        assert_eq!(tx.query_row("SELECT count(*) FROM network_deliveries WHERE case_id=?1",[&id],|r|r.get::<_,i64>(0))?,1);
        assert_eq!(tx.query_row("SELECT count(*) FROM origin_ban_jobs WHERE case_id=?1",[&id],|r|r.get::<_,i64>(0))?,0);
        assert_eq!(tx.query_row("SELECT count(*) FROM training_samples WHERE case_id=?1",[&id],|r|r.get::<_,i64>(0))?,0);
        tx.commit()?;Ok(())
    }).await.unwrap();
}
