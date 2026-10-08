use super::*;
use crate::operations::{Controls, Outcome, Patch};
use serde_json::Value;

pub(super) async fn set(runtime: &Runtime, new: bool, pending: bool, network: bool) -> Value {
    let Outcome::Ready(snapshot) = runtime.host_operations(HOST_ID).await.unwrap() else {
        panic!()
    };
    let patch = Patch {
        request_id: Uuid::new_v4().to_string(),
        expected_revision: snapshot["revision"].as_i64().unwrap(),
        controls: Controls {
            automatic_new_paused: new,
            automatic_pending_paused: pending,
            network_paused: network,
        },
    };
    let Outcome::Ready(v) = runtime.save_operations(HOST_ID, patch).await.unwrap() else {
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

#[tokio::test]
async fn controls_are_host_only_durable_and_replay_cannot_overwrite_later_changes() {
    let runtime = test_runtime().await;
    let patch = Patch {
        request_id: Uuid::new_v4().to_string(),
        expected_revision: 0,
        controls: Controls {
            automatic_new_paused: true,
            automatic_pending_paused: true,
            network_paused: true,
        },
    };
    assert!(matches!(
        runtime.host_operations(200).await.unwrap(),
        Outcome::Forbidden
    ));
    assert!(matches!(
        runtime.save_operations(200, patch.clone()).await.unwrap(),
        Outcome::Forbidden
    ));
    let Outcome::Ready(first) = runtime
        .save_operations(HOST_ID, patch.clone())
        .await
        .unwrap()
    else {
        panic!()
    };
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    assert!(
        restarted
            .operations_controls()
            .await
            .unwrap()
            .network_paused
    );
    set(&restarted, false, false, false).await;
    let Outcome::Ready(replayed) = restarted
        .save_operations(HOST_ID, patch.clone())
        .await
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(first, replayed);
    assert!(
        !restarted
            .operations_controls()
            .await
            .unwrap()
            .network_paused
    );
    let mut changed = patch.clone();
    changed.controls.network_paused = false;
    assert!(matches!(
        restarted.save_operations(HOST_ID, changed).await.unwrap(),
        Outcome::Conflict
    ));
    let mut stale = patch;
    stale.request_id = Uuid::new_v4().to_string();
    assert!(matches!(
        restarted.save_operations(HOST_ID, stale).await.unwrap(),
        Outcome::Conflict
    ));
    restarted
        .with_conn(|c| {
            assert_eq!(
                c.query_row("SELECT COUNT(*) FROM operations_requests", [], |r| r
                    .get::<_, i64>(0))?,
                2
            );
            assert_eq!(
                c.query_row(
                    "SELECT COUNT(*) FROM maintainer_actions WHERE command='緊急控制'",
                    [],
                    |r| r.get::<_, i64>(0)
                )?,
                2
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn admission_pause_blocks_automatic_bans_and_mutes_but_not_manual_requests() {
    let runtime = test_runtime().await;
    set(&runtime, true, false, false).await;
    let bot = TelegramStub::new(vec![]);
    for action in [
        ActionKind::AutoBan,
        ActionKind::GuestBotBan,
        ActionKind::GuestInvokerBan,
    ] {
        let case = dummy_case(action, -100, 200, Utc::now());
        assert!(!execute_auto_ban(&bot.bot, &runtime, case.clone(), "test")
            .await
            .unwrap());
        assert!(runtime.load_case(&case.id).await.unwrap().is_none());
    }
    for (index, action) in [ActionKind::FloodMute, ActionKind::CmdCleanMute]
        .into_iter()
        .enumerate()
    {
        let case = dummy_case(action, -100, 200, Utc::now());
        let id = runtime
            .queue_restriction(case, index as i32 + 1, None, "test")
            .await
            .unwrap();
        assert!(runtime.load_case(&id).await.unwrap().is_none());
    }
    let mut manual = dummy_case(ActionKind::SpamBan, -100, 201, Utc::now());
    manual.actor_user_id = Some(HOST_ID);
    manual.actor_name = Some("Host".into());
    let id = runtime.queue_manual_ban(manual, false, 9).await.unwrap();
    assert!(runtime.load_case(&id).await.unwrap().is_some());
    assert!(bot.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn paused_workers_do_not_claim_and_resume_rechecks_exemptions() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    runtime
        .queue_origin_bans(vec![(case.clone(), "test".into())])
        .await
        .unwrap();
    let restriction = dummy_case(ActionKind::FloodMute, -100, 201, Utc::now());
    runtime
        .queue_restriction(restriction.clone(), 2, None, "test")
        .await
        .unwrap();
    set(&runtime, false, true, false).await;
    let bot = TelegramStub::new(vec![]);
    assert!(
        !crate::origin_retry::attempt_origin_ban(&bot.bot, &runtime, case.clone())
            .await
            .unwrap()
    );
    crate::restriction_retry::attempt(&bot.bot, &runtime, restriction.clone())
        .await
        .unwrap();
    crate::origin_retry::retry_origin_bans(&bot.bot, &runtime)
        .await
        .unwrap();
    crate::restriction_retry::retry(&bot.bot, &runtime)
        .await
        .unwrap();
    assert!(bot.requests.lock().unwrap().is_empty());
    runtime
        .with_conn(|c| {
            assert_eq!(
                c.query_row("SELECT SUM(attempts) FROM origin_ban_jobs", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            assert_eq!(
                c.query_row("SELECT SUM(attempts) FROM restriction_jobs", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            Ok(())
        })
        .await
        .unwrap();
    runtime
        .set_group_whitelist(-100, 200, true, None)
        .await
        .unwrap();
    runtime
        .set_group_whitelist(-100, 201, true, None)
        .await
        .unwrap();
    set(&runtime, false, false, false).await;
    let now = Utc::now().timestamp();
    runtime.with_conn(move|c|{assert!(c.query_row("SELECT MIN(next_attempt_at) FROM origin_ban_jobs",[],|r|r.get::<_,i64>(0))?>now);c.execute_batch("UPDATE origin_ban_jobs SET next_attempt_at=0; UPDATE restriction_jobs SET next_attempt_at=0;")?;Ok(())}).await.unwrap();
    crate::origin_retry::retry_origin_bans(&bot.bot, &runtime)
        .await
        .unwrap();
    crate::restriction_retry::retry(&bot.bot, &runtime)
        .await
        .unwrap();
    assert_eq!(calls(&bot, "banchatmember"), 0);
    assert_eq!(calls(&bot, "restrictchatmember"), 0);
}

#[tokio::test]
async fn network_pause_covers_catchups_preserves_cooldown_and_allows_reversal() {
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
    set(&runtime, false, false, true).await;
    runtime
        .queue_network_catchup(&case.id, -300, 5, 200)
        .await
        .unwrap();
    let bot = TelegramStub::new(vec![]);
    assert_eq!(
        deliver_network_bans(&bot.bot, &runtime, None)
            .await
            .unwrap(),
        0
    );
    assert!(bot.requests.lock().unwrap().is_empty());
    runtime.delay_telegram_queue(120).await.unwrap();
    set(&runtime, false, false, false).await;
    runtime
        .with_conn(|c| {
            assert!(
                c.query_row(
                    "SELECT MAX(next_attempt_at)-MIN(next_attempt_at) FROM network_deliveries",
                    [],
                    |r| r.get::<_, i64>(0)
                )? >= 3
            );
            c.execute("UPDATE network_deliveries SET next_attempt_at=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
    assert_eq!(
        deliver_network_bans(&bot.bot, &runtime, None)
            .await
            .unwrap(),
        0
    );
    runtime
        .with_conn(|c| {
            c.execute("UPDATE telegram_retry_state SET not_before=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
    set(&runtime, true, true, true).await;
    reverse_ban_case(&bot.bot, &runtime, case.clone(), HOST_ID, "Host")
        .await
        .unwrap();
    assert_eq!(calls(&bot, "banchatmember"), 0);
    assert_eq!(calls(&bot, "unbanchatmember"), 1);
    set(&runtime, false, false, false).await;
    assert_eq!(
        deliver_network_bans(&bot.bot, &runtime, None)
            .await
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn receipt_failure_rolls_back_controls_and_concurrent_writes_have_one_winner() {
    let runtime = test_runtime().await;
    let patch = Patch {
        request_id: Uuid::new_v4().to_string(),
        expected_revision: 0,
        controls: Controls {
            automatic_new_paused: true,
            automatic_pending_paused: true,
            network_paused: true,
        },
    };
    runtime.with_conn(|c|{c.execute_batch("CREATE TRIGGER fail_controls BEFORE INSERT ON operations_requests BEGIN SELECT RAISE(ABORT,'injected'); END;")?;Ok(())}).await.unwrap();
    assert!(runtime
        .save_operations(HOST_ID, patch.clone())
        .await
        .is_err());
    assert!(
        !runtime
            .operations_controls()
            .await
            .unwrap()
            .automatic_new_paused
    );
    runtime
        .with_conn(|c| {
            assert_eq!(
                c.query_row("SELECT captcha_epoch FROM operations_controls", [], |r| r
                    .get::<_, i64>(
                    0
                ))?,
                0
            );
            assert_eq!(
                c.query_row("SELECT COUNT(*) FROM maintainer_actions", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            c.execute_batch("DROP TRIGGER fail_controls;")?;
            Ok(())
        })
        .await
        .unwrap();
    let mut other = patch.clone();
    other.request_id = Uuid::new_v4().to_string();
    let (a, b) = tokio::join!(
        runtime.save_operations(HOST_ID, patch),
        runtime.save_operations(HOST_ID, other)
    );
    assert!(matches!(
        (a.unwrap(), b.unwrap()),
        (Outcome::Ready(_), Outcome::Conflict) | (Outcome::Conflict, Outcome::Ready(_))
    ));
}

#[tokio::test]
async fn upgrade_restores_36_and_rejects_forged_fields() {
    let runtime = test_runtime().await;
    runtime.with_conn(|c|{c.execute_batch("DROP TABLE operations_requests; DROP TABLE operations_controls; PRAGMA user_version=36;")?;Ok(())}).await.unwrap();
    let result = crate::maintenance::check_upgrade(
        &runtime.config.sqlite_path,
        &runtime.config.data_dir.join("operations-upgrade"),
    )
    .unwrap();
    let result = serde_json::to_value(result).unwrap();
    assert_eq!(result["schema_before"], 36);
    assert_eq!(result["schema_after"], 38);
    assert_eq!(result["restore"], "ok");
    assert!(serde_json::from_value::<Patch>(serde_json::json!({"request_id":Uuid::new_v4().to_string(),"expected_revision":0,"actor_id":HOST_ID,"controls":{"automatic_new_paused":true,"automatic_pending_paused":false,"network_paused":false}})).is_err());
}
