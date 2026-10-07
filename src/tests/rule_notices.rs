use super::*;

#[tokio::test]
async fn cancelled_capture_keeps_rule_and_notice_in_the_same_commit() {
    let runtime = Arc::new(test_runtime().await);
    let (started, ready) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let blocked = runtime.clone();
    let blocker = tokio::spawn(async move {
        blocked
            .with_conn(move |_| {
                let _ = started.send(());
                let _ = wait.recv();
                Ok(())
            })
            .await
            .unwrap();
    });
    ready.await.unwrap();
    let saving = runtime.clone();
    let task = tokio::spawn(async move {
        saving.capture_rules(&bot_case("@CasinoBot")).await.unwrap();
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while runtime.spam_rules.try_write().is_ok() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    let held = runtime.spam_rules.try_read().is_err();
    release.send(()).unwrap();
    blocker.await.unwrap();
    assert!(held);
    assert_eq!(runtime.spam_rules.read().await.len(), 1);
    assert_eq!(counts(&runtime).await, (1, 1, 1, 0));
}

fn bot_case(text: &str) -> CaseRecord {
    let mut case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    case.evidence_text = text.into();
    case.matched_rule_pattern = Some("BOTSPAM".into());
    case
}

async fn counts(runtime: &Runtime) -> (i64, i64, i64, i64) {
    runtime
        .with_conn(|conn| {
            Ok((
                conn.query_row("SELECT COUNT(*) FROM spam_rules", [], |r| r.get(0))?,
                conn.query_row("SELECT COUNT(*) FROM rule_captures", [], |r| r.get(0))?,
                conn.query_row(
                    "SELECT COUNT(*) FROM rule_notice_jobs WHERE state='pending'",
                    [],
                    |r| r.get(0),
                )?,
                conn.query_row(
                    "SELECT COUNT(*) FROM rule_notice_jobs WHERE state='done'",
                    [],
                    |r| r.get(0),
                )?,
            ))
        })
        .await
        .unwrap()
}

async fn due(runtime: &Runtime) {
    runtime
        .with_conn(|conn| {
            conn.execute("UPDATE rule_notice_jobs SET next_attempt_at=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
}

fn sent(stub: &TelegramStub) -> Vec<serde_json::Value> {
    stub.requests
        .lock()
        .unwrap()
        .iter()
        .filter(|(method, _)| method == "sendmessage")
        .map(|(_, args)| args.clone())
        .collect()
}

#[tokio::test]
async fn failed_creation_notice_survives_restart_without_duplicate_rules() {
    let runtime = test_runtime().await;
    let case = bot_case("@CasinoBot");
    let failing = TelegramStub::with_failures(vec![], vec![("sendmessage".into(), -1)]);
    assert!(crate::rule_notices::capture(&failing.bot, &runtime, &case)
        .await
        .unwrap());
    assert_eq!(counts(&runtime).await, (1, 1, 1, 0));
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due(&restarted).await;
    let success = TelegramStub::new(vec![]);
    crate::rule_notices::retry(&success.bot, &restarted)
        .await
        .unwrap();
    crate::rule_notices::capture(&success.bot, &restarted, &case)
        .await
        .unwrap();
    assert_eq!(counts(&restarted).await, (1, 1, 0, 1));
    assert_eq!(sent(&success).len(), 1);
    assert!(sent(&success)[0]["text"]
        .as_str()
        .unwrap()
        .contains("@casinobot"));
}

#[tokio::test]
async fn concurrent_cases_share_one_rule_and_preserve_manual_rules() {
    let runtime = test_runtime().await;
    let first = bot_case("@CasinoBot @ManualBot");
    let second = bot_case("@casinobot");
    let manual = runtime
        .add_spam_rule("(?i)@ManualBot\\b", "manual")
        .await
        .unwrap();
    let (a, b, c) = tokio::join!(
        runtime.capture_rules(&first),
        runtime.capture_rules(&first),
        runtime.capture_rules(&second)
    );
    assert!(a.unwrap() && b.unwrap() && c.unwrap());
    assert_eq!(counts(&runtime).await, (2, 2, 1, 0));
    assert_eq!(
        runtime
            .spam_rules
            .read()
            .await
            .iter()
            .find(|r| r.id == manual)
            .unwrap()
            .description,
        "manual"
    );
    let stub = TelegramStub::new(vec![]);
    let (a, b) = tokio::join!(
        crate::rule_notices::retry(&stub.bot, &runtime),
        crate::rule_notices::retry(&stub.bot, &runtime)
    );
    a.unwrap();
    b.unwrap();
    assert_eq!(sent(&stub).len(), 1);
}

#[tokio::test]
async fn failed_outbox_insert_rolls_back_rules_receipt_and_cache() {
    let runtime = test_runtime().await;
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_notice BEFORE INSERT ON rule_notice_jobs BEGIN SELECT RAISE(ABORT,'test failure'); END;")?;
        Ok(())
    }).await.unwrap();
    let case = bot_case("@CasinoBot");
    assert!(runtime.capture_rules(&case).await.is_err());
    assert_eq!(counts(&runtime).await, (0, 0, 0, 0));
    assert!(runtime.spam_rules.read().await.is_empty());
    runtime
        .with_conn(|conn| {
            conn.execute_batch("DROP TRIGGER fail_notice")?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(runtime.capture_rules(&case).await.unwrap());
    assert_eq!(counts(&runtime).await, (1, 1, 1, 0));
}

#[tokio::test]
async fn old_capture_does_not_restore_deleted_rules_or_announce_edited_ones() {
    let runtime = test_runtime().await;
    let case = bot_case("@FirstBot @SecondBot");
    runtime.capture_rules(&case).await.unwrap();
    let ids = runtime
        .spam_rules
        .read()
        .await
        .iter()
        .map(|r| r.id)
        .collect::<Vec<_>>();
    runtime.delete_spam_rule(ids[0]).await.unwrap();
    runtime
        .update_spam_rule_pattern(ids[1], "changed")
        .await
        .unwrap();
    runtime.capture_rules(&case).await.unwrap();
    let stub = TelegramStub::new(vec![]);
    crate::rule_notices::retry(&stub.bot, &runtime)
        .await
        .unwrap();
    assert!(sent(&stub).is_empty());
    assert_eq!(counts(&runtime).await, (1, 1, 0, 0));
    runtime.capture_rules(&bot_case("@FirstBot")).await.unwrap();
    assert_eq!(counts(&runtime).await, (2, 2, 1, 0));
}

#[tokio::test]
async fn lost_send_acknowledgement_retries_notice_but_never_recreates_rule() {
    let runtime = test_runtime().await;
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_ack BEFORE UPDATE OF state ON rule_notice_jobs WHEN NEW.state='done' BEGIN SELECT RAISE(ABORT,'test acknowledgement failure'); END;")?;
        Ok(())
    }).await.unwrap();
    let case = bot_case("@CasinoBot");
    let stub = TelegramStub::new(vec![]);
    crate::rule_notices::capture(&stub.bot, &runtime, &case)
        .await
        .unwrap();
    assert_eq!(counts(&runtime).await, (1, 1, 1, 0));
    runtime
        .with_conn(|conn| {
            conn.execute_batch("DROP TRIGGER fail_ack")?;
            Ok(())
        })
        .await
        .unwrap();
    due(&runtime).await;
    crate::rule_notices::capture(&stub.bot, &runtime, &case)
        .await
        .unwrap();
    assert_eq!(counts(&runtime).await, (1, 1, 0, 1));
    // Telegram has no idempotency key for sendMessage; an unacknowledged send can repeat.
    assert_eq!(sent(&stub).len(), 2);
}

#[tokio::test]
async fn telegram_cooldown_survives_restart_and_notice_appears_in_queue() {
    let runtime = test_runtime().await;
    let limited = TelegramStub::with_api_errors(
        vec![],
        vec![(
            "sendmessage".into(),
            -1,
            serde_json::json!({"ok":false,"error_code":429,"description":"Too Many Requests","parameters":{"retry_after":120}}),
        )],
    );
    crate::rule_notices::capture(&limited.bot, &runtime, &bot_case("@CasinoBot"))
        .await
        .unwrap();
    assert!(runtime
        .queue_status(None)
        .await
        .unwrap()
        .contains("規則通知"));
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    due(&restarted).await;
    let stub = TelegramStub::new(vec![]);
    crate::rule_notices::retry(&stub.bot, &restarted)
        .await
        .unwrap();
    assert!(sent(&stub).is_empty());
    restarted
        .with_conn(|conn| {
            conn.execute("UPDATE telegram_retry_state SET not_before=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
    crate::rule_notices::retry(&stub.bot, &restarted)
        .await
        .unwrap();
    assert_eq!(sent(&stub).len(), 1);
}

#[tokio::test]
async fn large_capture_splits_notices_and_ignores_normal_conversation() {
    let runtime = test_runtime().await;
    assert!(!runtime
        .capture_rules(&bot_case("請看 @ExampleBot"))
        .await
        .unwrap());
    let text = (0..45)
        .map(|n| format!("@Example{n}Bot"))
        .collect::<Vec<_>>()
        .join(" ");
    runtime.capture_rules(&bot_case(&text)).await.unwrap();
    let stub = TelegramStub::new(vec![]);
    crate::rule_notices::retry(&stub.bot, &runtime)
        .await
        .unwrap();
    assert_eq!(counts(&runtime).await, (45, 1, 0, 3));
    assert_eq!(sent(&stub).len(), 3);
    for notice in sent(&stub) {
        let text = notice["text"].as_str().unwrap();
        assert!(text.chars().count() < 4096);
        assert!(text.matches("（規則 #").count() <= 20);
    }
}
