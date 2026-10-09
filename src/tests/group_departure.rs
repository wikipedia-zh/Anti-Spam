use super::*;
use crate::group_departure::{Outcome, Patch};
use serde_json::{json, Value};

async fn read(runtime: &Runtime, chat: i64) -> Value {
    let Outcome::Ready(v) = runtime.host_departure(HOST_ID, chat).await.unwrap() else {
        panic!()
    };
    v
}
async fn patch(runtime: &Runtime, chat: i64, block: bool) -> Patch {
    runtime.record_group_seen(chat, Some("Test group")).await;
    Patch {
        request_id: Uuid::new_v4().to_string(),
        chat_id: chat,
        expected_revision: read(runtime, chat).await["revision"]
            .as_str()
            .unwrap()
            .into(),
        reason: "沒有給予機器人足夠的權限".into(),
        block_rejoin: block,
    }
}
async fn queue(runtime: &Runtime, p: Patch) -> Value {
    let Outcome::Ready(v) = runtime.queue_departure(HOST_ID, p).await.unwrap() else {
        panic!()
    };
    v
}
fn calls(bot: &TelegramStub, method: &str) -> usize {
    bot.requests
        .lock()
        .unwrap()
        .iter()
        .filter(|(m, _)| m == method)
        .count()
}
async fn due(runtime: &Runtime) {
    runtime
        .with_conn(|conn| {
            conn.execute("UPDATE group_departures SET next_attempt_at=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn departure_is_host_only_protected_and_requires_current_confirmation() {
    let runtime = test_runtime().await;
    let p = patch(&runtime, -100, false).await;
    assert!(matches!(
        runtime.queue_departure(200, p.clone()).await.unwrap(),
        Outcome::Forbidden
    ));
    assert!(matches!(
        runtime.host_departure(200, -100).await.unwrap(),
        Outcome::Forbidden
    ));
    for chat in [
        1,
        runtime.config.log_channel_id,
        runtime.config.report_channel_id,
    ] {
        let p = patch(&runtime, chat, true).await;
        assert!(matches!(
            runtime.queue_departure(HOST_ID, p).await.unwrap(),
            Outcome::Forbidden
        ));
    }
    runtime.set_project_chat(-300).await;
    let protected = patch(&runtime, -300, true).await;
    assert!(matches!(
        runtime.queue_departure(HOST_ID, protected).await.unwrap(),
        Outcome::Forbidden
    ));
    let mut stale = p.clone();
    stale.expected_revision = "old".into();
    assert!(matches!(
        runtime.queue_departure(HOST_ID, stale).await.unwrap(),
        Outcome::Conflict
    ));
    let mut empty = p.clone();
    empty.reason = "  ".into();
    assert!(matches!(
        runtime.queue_departure(HOST_ID, empty).await.unwrap(),
        Outcome::Invalid
    ));
    let mut forged = serde_json::to_value(&p).unwrap();
    forged["actor_id"] = json!(HOST_ID);
    assert!(serde_json::from_value::<Patch>(forged).is_err());
    queue(&runtime, p.clone()).await;
    let mut changed = p.clone();
    changed.request_id = Uuid::new_v4().to_string();
    assert!(matches!(
        runtime.queue_departure(HOST_ID, changed).await.unwrap(),
        Outcome::Conflict
    ));
    assert!(!runtime.is_group_banned(-100).await);
}

#[tokio::test]
async fn duplicate_departure_survives_restart_and_never_leaves_twice() {
    let runtime = test_runtime().await;
    let p = patch(&runtime, -100, true).await;
    let (a, b) = tokio::join!(
        runtime.queue_departure(HOST_ID, p.clone()),
        runtime.queue_departure(HOST_ID, p.clone())
    );
    assert!(matches!(a.unwrap(), Outcome::Ready(_)));
    assert!(matches!(b.unwrap(), Outcome::Ready(_)));
    assert!(runtime.is_group_banned(-100).await);
    let runtime = Runtime::load(runtime.config.clone()).await.unwrap();
    runtime.me_id.set(UserId(999)).unwrap();
    let bot = TelegramStub::new(vec![]);
    crate::group_departure::retry(&bot.bot, &runtime)
        .await
        .unwrap();
    queue(&runtime, p.clone()).await;
    crate::group_departure::retry(&bot.bot, &runtime)
        .await
        .unwrap();
    assert_eq!(calls(&bot, "leavechat"), 1);
    assert_eq!(calls(&bot, "sendmessage"), 1);
    let data = read(&runtime, -100).await;
    assert_eq!(data["latest"]["state"], "done");
    assert_eq!(data["latest"]["notice_state"], "sent");
    runtime
        .set_group_banned(-100, false, "", Some(HOST_ID))
        .await
        .unwrap();
    queue(&runtime, p.clone()).await;
    assert!(
        !runtime.is_group_banned(-100).await,
        "an old request cannot reinstate a lifted service ban"
    );
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row(
                    "SELECT COUNT(*) FROM maintainer_actions WHERE command='退群'",
                    [],
                    |r| r.get::<_, i64>(0)
                )?,
                1
            );
            Ok(())
        })
        .await
        .unwrap();
    let mut changed = p;
    changed.reason = "Changed reason".into();
    assert!(matches!(
        runtime.queue_departure(HOST_ID, changed).await.unwrap(),
        Outcome::Conflict
    ));
}

#[tokio::test]
async fn failed_notice_does_not_prevent_departure_and_failed_leave_is_not_success() {
    let runtime = test_runtime().await;
    runtime.me_id.set(UserId(999)).unwrap();
    queue(&runtime, patch(&runtime, -100, false).await).await;
    let bot = TelegramStub::with_failures(
        vec![],
        vec![("sendmessage".into(), -100), ("leavechat".into(), -100)],
    );
    crate::group_departure::retry(&bot.bot, &runtime)
        .await
        .unwrap();
    let latest = read(&runtime, -100).await["latest"].clone();
    assert_eq!(latest["state"], "unconfirmed");
    assert_eq!(latest["notice_state"], "failed");
    assert!(!latest["last_error"].is_null());
    due(&runtime).await;
    crate::group_departure::retry(&bot.bot, &runtime)
        .await
        .unwrap();
    assert_eq!(read(&runtime, -100).await["latest"]["state"], "failed");
    assert_eq!(calls(&bot, "leavechat"), 1);
    let p = patch(&runtime, -100, false).await;
    queue(&runtime, p).await;
    let success = TelegramStub::new(vec![]);
    crate::group_departure::retry(&success.bot, &runtime)
        .await
        .unwrap();
    assert_eq!(read(&runtime, -100).await["latest"]["state"], "done");
    assert!(!runtime.is_group_banned(-100).await);
}

#[tokio::test]
async fn restart_after_uncertain_leave_checks_membership_without_sending_again() {
    let runtime = test_runtime().await;
    let p = patch(&runtime, -100, false).await;
    queue(&runtime, p).await;
    runtime
        .with_conn(|conn| {
            conn.execute(
                "UPDATE group_departures SET state='leaving',notice_state='unconfirmed'",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    let runtime = Runtime::load(runtime.config.clone()).await.unwrap();
    runtime.me_id.set(UserId(999)).unwrap();
    let bot = TelegramStub::with_members(vec![], vec![], true);
    bot.members.lock().unwrap().insert(
        (-100, 999),
        json!({"user":{"id":999,"is_bot":true,"first_name":"Bot"},"status":"left"}),
    );
    crate::group_departure::retry(&bot.bot, &runtime)
        .await
        .unwrap();
    assert_eq!(read(&runtime, -100).await["latest"]["state"], "done");
    assert_eq!(calls(&bot, "leavechat"), 0);
    assert_eq!(calls(&bot, "sendmessage"), 0);
}

#[tokio::test]
async fn lookup_failure_and_cooldown_do_not_trigger_leave() {
    let runtime = test_runtime().await;
    runtime.me_id.set(UserId(999)).unwrap();
    queue(&runtime, patch(&runtime, -100, false).await).await;
    let bot = TelegramStub::with_api_errors(
        vec![],
        vec![(
            "getchatmember".into(),
            -100,
            json!({"ok":false,"error_code":429,"description":"Too Many Requests","parameters":{"retry_after":120}}),
        )],
    );
    crate::group_departure::retry(&bot.bot, &runtime)
        .await
        .unwrap();
    assert_eq!(read(&runtime, -100).await["latest"]["state"], "queued");
    assert_eq!(calls(&bot, "leavechat"), 0);
    due(&runtime).await;
    crate::group_departure::retry(&bot.bot, &runtime)
        .await
        .unwrap();
    assert_eq!(calls(&bot, "getchatmember"), 1);
    runtime
        .with_conn(|conn| {
            assert!(
                conn.query_row("SELECT not_before FROM telegram_retry_state", [], |r| r
                    .get::<_, i64>(0))?
                    > Utc::now().timestamp()
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn receipt_failure_rolls_back_group_ban_audit_and_cache() {
    let runtime = test_runtime().await;
    let p = patch(&runtime, -100, true).await;
    runtime.with_conn(|conn|{conn.execute_batch("CREATE TRIGGER fail_departure BEFORE INSERT ON group_departures BEGIN SELECT RAISE(ABORT,'fail'); END;")?;Ok(())}).await.unwrap();
    assert!(runtime.queue_departure(HOST_ID, p).await.is_err());
    assert!(!runtime.is_group_banned(-100).await);
    runtime
        .with_conn(|conn| {
            for table in ["banned_groups", "maintainer_actions", "group_departures"] {
                assert_eq!(
                    conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r
                        .get::<_, i64>(0))?,
                    0
                );
            }
            Ok(())
        })
        .await
        .unwrap();
}

#[test]
fn departure_migration_preserves_data_and_restores() {
    let dir = std::env::temp_dir().join(format!("spb-departure-{}", Uuid::new_v4()));
    std::fs::create_dir(&dir).unwrap();
    let db = dir.join("bot.db");
    let mut conn = Connection::open(&db).unwrap();
    Runtime::init_db(&mut conn).unwrap();
    conn.execute_batch("DROP TABLE group_departures;PRAGMA user_version=37;")
        .unwrap();
    drop(conn);
    let result =
        serde_json::to_value(crate::maintenance::check_upgrade(&db, &dir.join("check")).unwrap())
            .unwrap();
    assert_eq!(result["schema_before"], 37);
    assert_eq!(result["schema_after"], 41);
    assert_eq!(result["restore"], "ok");
}
