use super::*;
use serde_json::json;

fn membership(status: &str, date: i64) -> ChatMemberUpdated {
    serde_json::from_value(
        json!({"chat":{"id":-300,"type":"supergroup","title":"Test group"},
        "from":{"id":99,"is_bot":false,"first_name":"Admin"},"date":date,
        "old_chat_member":{"user":{"id":999,"is_bot":true,"first_name":"Bot"},"status":"member"},
        "new_chat_member":{"user":{"id":999,"is_bot":true,"first_name":"Bot"},"status":status}}),
    )
    .unwrap()
}
async fn state(runtime: &Runtime) -> (String, i64) {
    runtime
        .with_conn(|conn| crate::group_access::snapshot(conn, -300))
        .await
        .unwrap()
}
async fn queued() -> (Runtime, CaseRecord) {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    for chat in [-300, -400] {
        runtime
            .set_group_module(chat, "netban", true)
            .await
            .unwrap();
    }
    runtime.enqueue_network_deliveries(&case.id).await.unwrap();
    (runtime, case)
}
fn missing(method: &str) -> TelegramStub {
    TelegramStub::with_api_errors(
        vec![],
        vec![(
            method.into(),
            -300,
            json!({"ok":false,"error_code":400,"description":"Bad Request: chat not found"}),
        )],
    )
}
async fn due(runtime: &Runtime) {
    runtime.with_conn(|c|{c.execute_batch("UPDATE network_deliveries SET next_attempt_at=0;UPDATE network_catchups SET next_attempt_at=0;")?;Ok(())}).await.unwrap();
}

#[tokio::test]
async fn missing_chat_holds_all_its_deliveries_without_blocking_other_groups() {
    for method in ["getchatmember", "banchatmember"] {
        let (runtime, case) = queued().await;
        let bot = missing(method);
        deliver_network_bans(&bot.bot, &runtime, None)
            .await
            .unwrap();
        assert_eq!(state(&runtime).await.0, "unavailable");
        assert_eq!(
            runtime.list_network_ban_targets(&case.id).await.unwrap(),
            vec![-400]
        );
        let restart = Runtime::load(runtime.config.clone()).await.unwrap();
        due(&restart).await;
        let good = TelegramStub::new(vec![]);
        assert_eq!(
            deliver_network_bans(&good.bot, &restart, None)
                .await
                .unwrap(),
            0
        );
        assert!(good.requests.lock().unwrap().is_empty());
        assert!(restart
            .queue_status(Some(&case.id))
            .await
            .unwrap()
            .contains("等待機器人重新加入"));
        let page = restart
            .host_query(crate::host_panel::Query {
                view: "queue".into(),
                search: case.id.clone(),
                offset: 0,
                filter: String::new(),
                created_from: None,
                created_before: None,
            })
            .await
            .unwrap();
        assert_eq!(page["items"][0]["waiting_for_group"], 1);
        assert_eq!(page["items"][0]["membership_state"], "unavailable");
        let new = dummy_case(ActionKind::AutoBan, -100, 201, Utc::now());
        restart.persist_case(&new).await.unwrap();
        restart.enqueue_network_deliveries(&new.id).await.unwrap();
        let id = new.id.clone();
        restart
            .with_conn(move |c| {
                assert_eq!(
                    c.query_row(
                        "SELECT COUNT(*) FROM network_deliveries WHERE case_id=?1 AND chat_id=-300",
                        [id],
                        |r| r.get::<_, i64>(0)
                    )?,
                    0
                );
                Ok(())
            })
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn rejoin_ignores_old_events_and_failures_and_rechecks_exemptions() {
    let (runtime, _case) = queued().await;
    let now = Utc::now().timestamp();
    runtime
        .record_bot_membership(10, membership("left", now))
        .await
        .unwrap();
    let old_revision = runtime.group_access_revision(-300).await.unwrap();
    runtime
        .record_bot_membership(11, membership("member", now))
        .await
        .unwrap();
    let expected = state(&runtime).await;
    runtime
        .record_bot_membership(10, membership("left", now))
        .await
        .unwrap();
    runtime
        .record_bot_membership(11, membership("member", now))
        .await
        .unwrap();
    runtime
        .record_group_access_error(
            -300,
            old_revision,
            &teloxide::RequestError::Api(teloxide::ApiError::ChatNotFound),
        )
        .await
        .unwrap();
    assert_eq!(state(&runtime).await, expected);
    runtime
        .with_conn(|c| {
            c.execute(
                "INSERT INTO group_whitelist(chat_id,user_id,created_at) VALUES (-300,200,'now')",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    due(&runtime).await;
    let bot = TelegramStub::new(vec![]);
    deliver_network_bans(&bot.bot, &runtime, None)
        .await
        .unwrap();
    assert!(!bot
        .requests
        .lock()
        .unwrap()
        .iter()
        .any(|(_, args)| args["chat_id"] == -300));
    assert_eq!(state(&runtime).await.0, "present");
}

#[tokio::test]
async fn missing_chat_does_not_erase_uncertain_ban_or_reversal_work() {
    let (runtime, case) = queued().await;
    runtime
        .with_conn(|c| {
            c.execute(
                "UPDATE network_deliveries SET outcome_unknown=1 WHERE chat_id=-300",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    let bot = missing("getchatmember");
    deliver_network_bans(&bot.bot, &runtime, None)
        .await
        .unwrap();
    let id = case.id.clone();
    runtime
        .with_conn(move |c| {
            assert!(c.query_row(
                "SELECT outcome_unknown FROM network_deliveries WHERE case_id=?1 AND chat_id=-300",
                [id],
                |r| r.get::<_, bool>(0)
            )?);
            Ok(())
        })
        .await
        .unwrap();
    let failed = TelegramStub::with_failures(vec![], vec![("unbanchatmember".into(), -300)]);
    assert!(
        reverse_ban_case(&failed.bot, &runtime, case.clone(), HOST_ID, "Host")
            .await
            .is_err()
    );
    let id = case.id.clone();
    runtime
        .with_conn(move |c| {
            assert_eq!(
                c.query_row("SELECT status FROM cases WHERE id=?1", [id], |r| r
                    .get::<_, String>(0))?,
                "reversal_pending"
            );
            Ok(())
        })
        .await
        .unwrap();
    let revision = runtime.group_access_revision(-300).await.unwrap();
    runtime
        .record_group_access_check(-300, revision, "present")
        .await
        .unwrap();
    due(&runtime).await;
    let good = TelegramStub::new(vec![]);
    deliver_network_bans(&good.bot, &runtime, None)
        .await
        .unwrap();
    assert!(good.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn membership_checks_recover_old_errors_once_and_respect_cooldown() {
    let (runtime, _) = queued().await;
    runtime.me_id.set(UserId(999)).unwrap();
    runtime
        .with_conn(|c| {
            c.execute(
                "UPDATE network_deliveries SET last_error='old failure' WHERE chat_id=-300",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    let bad = missing("getchatmember");
    crate::group_access::reconcile(&bad.bot, &runtime)
        .await
        .unwrap();
    assert_eq!(state(&runtime).await.0, "unavailable");
    let count = bad.requests.lock().unwrap().len();
    crate::group_access::reconcile(&bad.bot, &runtime)
        .await
        .unwrap();
    assert_eq!(bad.requests.lock().unwrap().len(), count);
    runtime
        .with_conn(|c| {
            c.execute("UPDATE group_access SET next_check_at=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
    runtime.delay_telegram_queue(120).await.unwrap();
    let good = TelegramStub::new(vec![]);
    crate::group_access::reconcile(&good.bot, &runtime)
        .await
        .unwrap();
    assert!(!good
        .requests
        .lock()
        .unwrap()
        .iter()
        .any(|(method, _)| method == "getchatmember"));
    runtime
        .with_conn(|c| {
            c.execute("UPDATE telegram_retry_state SET not_before=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
    crate::group_access::reconcile(&good.bot, &runtime)
        .await
        .unwrap();
    assert_eq!(state(&runtime).await.0, "present");
}

#[tokio::test]
async fn permissions_and_transient_errors_do_not_mark_a_group_removed() {
    let runtime = test_runtime().await;
    for error in [
        teloxide::ApiError::Unknown("Bad Request: not enough rights".into()),
        teloxide::ApiError::Unknown("Internal Server Error".into()),
    ] {
        runtime
            .record_group_access_error(
                -300,
                runtime.group_access_revision(-300).await.unwrap(),
                &teloxide::RequestError::Api(error),
            )
            .await
            .unwrap();
        assert_eq!(state(&runtime).await, ("unknown".into(), 0));
    }
    runtime
        .record_group_access_error(
            -300,
            runtime.group_access_revision(-300).await.unwrap(),
            &teloxide::RequestError::Api(teloxide::ApiError::BotKickedFromSupergroup),
        )
        .await
        .unwrap();
    assert_eq!(state(&runtime).await.0, "left");
}

#[tokio::test]
async fn catchup_waits_for_access_without_repeating_the_completed_ban() {
    let (runtime, case) = queued().await;
    runtime
        .queue_network_catchup(&case.id, -300, 42, 200)
        .await
        .unwrap();
    let good = TelegramStub::new(vec![]);
    deliver_network_bans(&good.bot, &runtime, None)
        .await
        .unwrap();
    let bad = missing("deletemessage");
    crate::network_catchup::retry(&bad.bot, &runtime)
        .await
        .unwrap();
    assert_eq!(state(&runtime).await.0, "unavailable");
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due(&restarted).await;
    let before = good.requests.lock().unwrap().len();
    crate::network_catchup::retry(&good.bot, &restarted)
        .await
        .unwrap();
    assert_eq!(good.requests.lock().unwrap().len(), before);
    let revision = restarted.group_access_revision(-300).await.unwrap();
    restarted
        .record_group_access_check(-300, revision, "present")
        .await
        .unwrap();
    due(&restarted).await;
    crate::network_catchup::retry(&good.bot, &restarted)
        .await
        .unwrap();
    let calls = good.requests.lock().unwrap();
    assert_eq!(
        calls
            .iter()
            .filter(|(method, args)| method == "banchatmember" && args["chat_id"] == -300)
            .count(),
        1
    );
    assert_eq!(
        calls
            .iter()
            .filter(|(method, args)| method == "deletemessage" && args["chat_id"] == -300)
            .count(),
        1
    );
}

#[test]
fn access_migration_preserves_existing_data_and_restores() {
    let dir = std::env::temp_dir().join(format!("spb-access-{}", Uuid::new_v4()));
    std::fs::create_dir(&dir).unwrap();
    let db = dir.join("bot.db");
    let mut conn = Connection::open(&db).unwrap();
    Runtime::init_db(&mut conn).unwrap();
    conn.execute_batch("DROP TABLE group_access;PRAGMA user_version=38;")
        .unwrap();
    drop(conn);
    let result =
        serde_json::to_value(crate::maintenance::check_upgrade(&db, &dir.join("check")).unwrap())
            .unwrap();
    assert_eq!(result["schema_before"], 38);
    assert_eq!(result["schema_after"], 39);
    assert_eq!(result["restore"], "ok");
}
