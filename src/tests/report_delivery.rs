use super::*;
use crate::report_delivery::{self, Submission};

fn report_case() -> CaseRecord {
    let mut case = dummy_case(ActionKind::PendingReport, -100, 200, Utc::now());
    case.status = "pending_review".into();
    case.actor_user_id = Some(444);
    case.source_message_id = Some(1);
    case.evidence_text = "casino gambling".into();
    case
}

fn calls(telegram: &TelegramStub, method: &str) -> Vec<serde_json::Value> {
    telegram
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter(|(m, _)| m == method)
        .map(|(_, v)| v.clone())
        .collect()
}

async fn due(runtime: &Runtime) {
    runtime.with_conn(|conn| {
        conn.execute_batch("UPDATE report_deliveries SET next_attempt_at=0; UPDATE review_updates SET next_attempt_at=0; UPDATE telegram_retry_state SET not_before=0;")?;
        Ok(())
    }).await.unwrap();
}

async fn enqueue(runtime: &Runtime, case: CaseRecord) -> String {
    let Submission::Queued(id) = runtime.queue_report(case, 99).await.unwrap() else {
        panic!("report suspended");
    };
    id
}

#[tokio::test]
async fn failed_report_card_survives_restart_and_command_replay() {
    let runtime = test_runtime().await;
    let case = report_case();
    let id = enqueue(&runtime, case.clone()).await;
    let failed = TelegramStub::with_failures(vec![], vec![("sendmessage".into(), -1)]);
    report_delivery::deliver(&failed.bot, &runtime, &id)
        .await
        .unwrap();
    assert_eq!(calls(&failed, "sendmessage").len(), 1);
    assert!(runtime
        .queue_status(None)
        .await
        .unwrap()
        .contains("舉報送出：1"));
    assert_eq!(
        runtime
            .load_case(&id)
            .await
            .unwrap()
            .unwrap()
            .log_message_id,
        None
    );
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let mut replay = case;
    replay.id = Uuid::new_v4().to_string();
    assert_eq!(enqueue(&restarted, replay).await, id);
    due(&restarted).await;
    let success = TelegramStub::new(vec![]);
    report_delivery::retry(&success.bot, &restarted)
        .await
        .unwrap();
    let sent = calls(&success, "sendmessage");
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0]["chat_id"], -1);
    assert!(sent[0]["reply_markup"].to_string().contains(&id));
    assert_eq!(sent[1]["chat_id"], -100);
    assert_eq!(sent[1]["reply_parameters"]["message_id"], 99);
    assert_eq!(
        sent[1]["reply_parameters"]["allow_sending_without_reply"],
        true
    );
    assert!(restarted
        .queue_status(None)
        .await
        .unwrap()
        .contains("未完成：0 · 待人工審核：1"));
    due(&restarted).await;
    report_delivery::retry(&success.bot, &restarted)
        .await
        .unwrap();
    assert_eq!(calls(&success, "sendmessage").len(), 2);
}

#[tokio::test]
async fn failed_report_confirmation_does_not_resend_review_card() {
    let runtime = test_runtime().await;
    let id = enqueue(&runtime, report_case()).await;
    let failed = TelegramStub::with_failures(vec![], vec![("sendmessage".into(), -100)]);
    report_delivery::deliver(&failed.bot, &runtime, &id)
        .await
        .unwrap();
    assert_eq!(calls(&failed, "sendmessage").len(), 2);
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due(&restarted).await;
    let success = TelegramStub::new(vec![]);
    report_delivery::retry(&success.bot, &restarted)
        .await
        .unwrap();
    let sent = calls(&success, "sendmessage");
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0]["chat_id"], -100);
}

#[tokio::test]
async fn review_before_confirmation_updates_the_late_reply_after_restart() {
    let runtime = test_runtime().await;
    let case = report_case();
    enqueue(&runtime, case.clone()).await;
    let failed = TelegramStub::with_failures(vec![], vec![("sendmessage".into(), -100)]);
    report_delivery::deliver(&failed.bot, &runtime, &case.id)
        .await
        .unwrap();
    let guard = runtime.review_guard(&case.id).await;
    runtime
        .decide_report(&case, "reject", (HOST_ID, "Host".into()), (-1, 100), guard)
        .await
        .unwrap();
    let success = TelegramStub::new(vec![]);
    crate::moderation_queue::deliver_review_updates(&success.bot, &runtime, None)
        .await
        .unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due(&restarted).await;
    report_delivery::retry(&success.bot, &restarted)
        .await
        .unwrap();
    crate::moderation_queue::deliver_review_updates(&success.bot, &restarted, None)
        .await
        .unwrap();
    let sent = calls(&success, "sendmessage");
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0]["chat_id"], -100);
    assert!(!sent[0]["text"].as_str().unwrap().contains("審核。"));
    assert!(calls(&success, "editmessagetext")
        .iter()
        .any(|v| v["chat_id"] == -100 && v["text"] == "此舉報未被受理。"));
    assert_eq!(restarted.report_strikes(444).await, 1);
    assert_eq!(restarted.model.lock().await.ham_docs, 1);
}

#[tokio::test]
async fn report_queue_failure_rolls_back_case_and_does_not_send_a_review() {
    let runtime = test_runtime().await;
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_report BEFORE INSERT ON report_deliveries BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    report_delivery::handle(
        &telegram.bot,
        &runtime,
        &spam_ban_message(444, "/spam", Some("casino gambling")),
    )
    .await
    .unwrap();
    let sent = calls(&telegram, "sendmessage");
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0]["chat_id"], -100);
    assert!(sent[0]["text"].as_str().unwrap().contains("未能保存"));
    runtime
        .with_conn(|conn| {
            for table in ["cases", "moderation_requests", "report_deliveries"] {
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

#[tokio::test]
async fn lost_card_acknowledgement_does_not_reopen_a_decided_report() {
    let runtime = test_runtime().await;
    let case = report_case();
    enqueue(&runtime, case.clone()).await;
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_report_ack BEFORE UPDATE OF review_message_id ON report_deliveries BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    let telegram = TelegramStub::new(vec![]);
    assert!(report_delivery::deliver(&telegram.bot, &runtime, &case.id)
        .await
        .is_err());
    assert_eq!(calls(&telegram, "sendmessage").len(), 1);
    let guard = runtime.review_guard(&case.id).await;
    runtime
        .decide_report(&case, "reject", (HOST_ID, "Host".into()), (-1, 100), guard)
        .await
        .unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due(&restarted).await;
    report_delivery::retry(&telegram.bot, &restarted)
        .await
        .unwrap();
    let sent = calls(&telegram, "sendmessage");
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[1]["chat_id"], -100);
    assert!(restarted
        .queue_status(None)
        .await
        .unwrap()
        .contains("待人工審核：0"));
    assert_eq!(restarted.model.lock().await.ham_docs, 1);
}

#[tokio::test]
async fn concurrent_report_delivery_sends_one_card_and_confirmation() {
    let runtime = test_runtime().await;
    let id = enqueue(&runtime, report_case()).await;
    let telegram = TelegramStub::new(vec![]);
    let (a, b) = tokio::join!(
        report_delivery::deliver(&telegram.bot, &runtime, &id),
        report_delivery::deliver(&telegram.bot, &runtime, &id)
    );
    a.unwrap();
    b.unwrap();
    assert_eq!(calls(&telegram, "sendmessage").len(), 2);
}

#[tokio::test]
async fn report_rate_limit_is_shared_and_survives_restart() {
    let runtime = test_runtime().await;
    let id = enqueue(&runtime, report_case()).await;
    let limited = TelegramStub::with_api_errors(
        vec![],
        vec![(
            "sendmessage".into(),
            -1,
            serde_json::json!({"ok":false,"error_code":429,"description":"Too Many Requests","parameters":{"retry_after":120}}),
        )],
    );
    report_delivery::deliver(&limited.bot, &runtime, &id)
        .await
        .unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    restarted
        .with_conn(|conn| {
            conn.execute("UPDATE report_deliveries SET next_attempt_at=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
    let success = TelegramStub::new(vec![]);
    report_delivery::retry(&success.bot, &restarted)
        .await
        .unwrap();
    assert!(calls(&success, "sendmessage").is_empty());
    let status = restarted.queue_status(None).await.unwrap();
    assert!(status.contains("Telegram 限流"));
    assert!(status.contains("待執行 0 · 有錯誤 1"));
    due(&restarted).await;
    report_delivery::retry(&success.bot, &restarted)
        .await
        .unwrap();
    assert_eq!(calls(&success, "sendmessage").len(), 2);
}

#[tokio::test]
async fn report_submission_checks_strikes_atomically_but_replay_still_resumes() {
    let runtime = test_runtime().await;
    let id = enqueue(&runtime, report_case()).await;
    runtime.with_conn(|conn| {
        for user in [444,HOST_ID] {
            conn.execute("INSERT INTO report_offenses(user_id,rejected_count,last_rejected_at) VALUES (?1,3,'now')",[user])?;
        }
        Ok(())
    }).await.unwrap();
    assert_eq!(enqueue(&runtime, report_case()).await, id);
    assert!(matches!(
        runtime.queue_report(report_case(), 100).await.unwrap(),
        Submission::Suspended(3)
    ));
    let mut exempt = report_case();
    exempt.actor_user_id = Some(HOST_ID);
    assert!(matches!(
        runtime.queue_report(exempt, 100).await.unwrap(),
        Submission::Queued(_)
    ));
}
