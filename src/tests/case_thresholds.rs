use super::*;
use crate::origin_retry::{execute_scored_ban, retry_origin_bans};
use serde_json::Value;

fn scored(user: i64, score: f64) -> CaseRecord {
    let mut case = dummy_case(ActionKind::AutoBan, -100, user, Utc::now());
    case.model_score = Some(score);
    case.matched_rule_pattern = Some("ML".into());
    case
}
async fn preview(runtime: &Runtime, id: &str) -> Value {
    runtime
        .host_case(crate::host_cases::Query {
            case_id: id.into(),
            offset: 0,
        })
        .await
        .unwrap()
        .unwrap()
}
async fn due(runtime: &Runtime) {
    runtime.with_conn(|conn| {
        conn.execute_batch("UPDATE origin_ban_jobs SET next_attempt_at=0; UPDATE telegram_retry_state SET not_before=0;")?;
        Ok(())
    }).await.unwrap();
}

#[tokio::test]
async fn changed_local_threshold_cancels_a_queued_ban_without_rewriting_detection() {
    let runtime = test_runtime().await;
    runtime.set_threshold(0.9).await.unwrap();
    runtime.set_group_threshold(-100, Some(0.6)).await.unwrap();
    let case = scored(200, 0.7);
    runtime
        .queue_scored_ban(case.clone(), "test".into(), 0.6)
        .await
        .unwrap();
    runtime
        .queue_scored_ban(case.clone(), "replay".into(), 0.65)
        .await
        .unwrap();
    let before = preview(&runtime, &case.id).await;
    assert_eq!(before["threshold_checks"].as_array().unwrap().len(), 1);
    assert_eq!(before["threshold_checks"][0]["threshold"], 0.6);
    runtime.set_group_threshold(-100, Some(0.8)).await.unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let bot = TelegramStub::new(vec![]);
    retry_origin_bans(&bot.bot, &restarted).await.unwrap();
    assert!(bot
        .requests
        .lock()
        .unwrap()
        .iter()
        .all(|(m, _)| m != "banchatmember"));
    let after = preview(&restarted, &case.id).await;
    assert_eq!(after["case"]["status"], "ban_failed");
    assert_eq!(after["threshold_checks"][0], before["threshold_checks"][0]);
    assert_eq!(after["threshold_checks"][1]["phase"], "enforcement");
    assert_eq!(after["threshold_checks"][1]["threshold"], 0.8);
    assert_eq!(after["threshold_checks"][1]["passed"], false);
    assert_eq!(after["threshold_checks"].as_array().unwrap().len(), 2);
    assert_ne!(before["revision"], after["revision"]);
}

#[tokio::test]
async fn local_and_network_checks_record_different_thresholds_and_survive_setting_changes() {
    let runtime = test_runtime().await;
    runtime.set_threshold(0.9).await.unwrap();
    runtime.set_group_threshold(-100, Some(0.6)).await.unwrap();
    let case = scored(200, 0.7);
    let bot = TelegramStub::new(vec![]);
    assert!(
        execute_scored_ban(&bot.bot, &runtime, case.clone(), "test", 0.6)
            .await
            .unwrap()
    );
    let before = preview(&runtime, &case.id).await;
    let checks = &before["threshold_checks"];
    assert_eq!(checks.as_array().unwrap().len(), 3);
    assert_eq!(checks[0]["threshold"], 0.6);
    assert_eq!(checks[1]["threshold"], 0.6);
    assert_eq!(checks[2]["threshold"], 0.9);
    assert_eq!(checks[2]["passed"], false);
    assert_eq!(before["case"]["netban_eligible"], false);
    runtime.set_threshold(0.65).await.unwrap();
    runtime.set_group_threshold(-100, Some(0.99)).await.unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    assert_eq!(
        preview(&restarted, &case.id).await["threshold_checks"],
        *checks
    );
    assert!(restarted
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn shared_ban_keeps_its_admission_threshold_through_retries_and_reversal() {
    let runtime = test_runtime().await;
    runtime.set_threshold(0.9).await.unwrap();
    runtime.set_group_threshold(-100, Some(0.6)).await.unwrap();
    let case = scored(200, 0.95);
    let failed = TelegramStub::with_failures(vec![], vec![("sendmessage".into(), -1)]);
    assert!(
        execute_scored_ban(&failed.bot, &runtime, case.clone(), "test", 0.6)
            .await
            .unwrap()
    );
    let before = preview(&runtime, &case.id).await;
    assert_eq!(before["case"]["netban_eligible"], true);
    assert_eq!(before["threshold_checks"][2]["threshold"], 0.9);
    assert_eq!(before["threshold_checks"][2]["passed"], true);
    runtime.set_threshold(0.99).await.unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let bot = TelegramStub::new(vec![]);
    due(&restarted).await;
    retry_origin_bans(&bot.bot, &restarted).await.unwrap();
    let after = preview(&restarted, &case.id).await;
    assert_eq!(after["threshold_checks"], before["threshold_checks"]);
    let request = crate::host_cases::Reverse {
        request_id: Uuid::new_v4().to_string(),
        case_id: case.id.clone(),
        target_user_id: 200,
        expected_revision: after["revision"].as_str().unwrap().into(),
    };
    assert!(matches!(
        restarted.reverse_host_case(HOST_ID, request).await.unwrap(),
        crate::host_cases::Outcome::Saved(_)
    ));
    assert_eq!(
        preview(&restarted, &case.id).await["threshold_checks"],
        before["threshold_checks"]
    );
    assert!(restarted
        .find_active_network_ban(200)
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn snapshot_write_failure_rolls_back_ban_intent_and_legacy_cases_stay_unknown() {
    let runtime = test_runtime().await;
    let legacy = scored(201, 0.7);
    runtime.persist_case(&legacy).await.unwrap();
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_snapshot BEFORE INSERT ON case_threshold_checks BEGIN SELECT RAISE(ABORT,'test failure'); END;")?;
        Ok(())
    }).await.unwrap();
    let case = scored(200, 0.7);
    assert!(runtime
        .queue_scored_ban(case.clone(), "test".into(), 0.6)
        .await
        .is_err());
    assert!(runtime.load_case(&case.id).await.unwrap().is_none());
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM origin_ban_jobs", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            conn.execute_batch("DROP TABLE case_threshold_checks; PRAGMA user_version=32;")?;
            Ok(())
        })
        .await
        .unwrap();
    let result = crate::maintenance::check_upgrade(
        &runtime.config.sqlite_path,
        &runtime.config.data_dir.join("threshold-upgrade"),
    )
    .unwrap();
    let result = serde_json::to_value(result).unwrap();
    assert_eq!(result["schema_before"], 32);
    assert_eq!(result["schema_after"], 38);
    assert_eq!(result["restore"], "ok");
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    assert!(preview(&restarted, &legacy.id).await["threshold_checks"]
        .as_array()
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn invalid_global_threshold_is_recorded_without_losing_the_local_ban_result() {
    let runtime = test_runtime().await;
    runtime.set_threshold(f64::INFINITY).await.unwrap();
    runtime.set_group_threshold(-100, Some(0.6)).await.unwrap();
    let case = scored(200, 0.7);
    let bot = TelegramStub::new(vec![]);
    assert!(
        execute_scored_ban(&bot.bot, &runtime, case.clone(), "test", 0.6)
            .await
            .unwrap()
    );
    let value = preview(&runtime, &case.id).await;
    assert_ne!(value["case"]["status"], "ban_pending");
    assert_eq!(value["threshold_checks"][2]["threshold"], Value::Null);
    assert_eq!(value["threshold_checks"][2]["passed"], false);
    assert_eq!(value["case"]["netban_eligible"], false);
}
