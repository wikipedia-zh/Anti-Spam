use super::*;
use crate::notices::{action_log, diagnostic, group_notice, HealthFailures};

fn rendered_length(html: &str) -> usize {
    let tags = StdRegex::new(r"<[^>]+>").unwrap();
    tags.replace_all(html, "")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .encode_utf16()
        .count()
}

#[tokio::test]
async fn long_evidence_is_clipped_in_notices_but_preserved_in_storage() {
    let runtime = test_runtime().await;
    let telegram = TelegramStub::new(vec![]);
    let mut case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    case.target_name = "<b>name & 😀</b>".repeat(100);
    case.actor_name = Some(case.target_name.clone());
    case.evidence_text = "<b>evidence & 😀</b>".repeat(1000);
    case.matched_rule_pattern = Some("<rule & 😀>".repeat(1000));
    runtime.persist_case(&case).await.unwrap();
    log_action(&telegram.bot, &runtime, &case).await.unwrap();
    queue_training_review(&telegram.bot, &runtime, &case).await.unwrap();
    let requests = telegram.requests.lock().unwrap().clone();
    let cards: Vec<_> = requests
        .iter()
        .filter(|(method, _)| method == "sendmessage")
        .collect();
    assert_eq!(cards.len(), 2);
    for (_, args) in cards {
        let text = args["text"].as_str().unwrap();
        assert!(rendered_length(text) <= 4096);
        assert!(text.contains("完整內容已保存在案例記錄"));
        assert!(text.contains("&lt;b&gt;evidence &amp; 😀&lt;/b&gt;"));
        assert!(!text.contains("<b>evidence"));
        assert!(text.contains(&case.id));
    }
    let lookup = format_case_lookup(&case, "-", "-");
    assert!(rendered_length(&lookup) <= 4096);
    assert!(!lookup.contains("href=\"-\""));
    assert_eq!(
        runtime
            .load_case(&case.id)
            .await
            .unwrap()
            .unwrap()
            .evidence_text,
        case.evidence_text
    );
}

#[tokio::test]
async fn a_failed_log_send_does_not_create_a_broken_group_link() {
    let runtime = test_runtime().await;
    let telegram = TelegramStub::new(vec![]);
    let case = dummy_case(ActionKind::Mute, -100, 200, Utc::now());
    notify_group(&telegram.bot, &runtime, &case, 0, "<b>已執行管理操作</b>")
        .await
        .unwrap();
    let requests = telegram.requests.lock().unwrap();
    let (_, args) = requests
        .iter()
        .find(|(method, _)| method == "sendmessage")
        .unwrap();
    let text = args["text"].as_str().unwrap();
    assert!(text.starts_with("<b>禁言</b>"));
    assert!(text.contains(&case.id));
    assert!(!text.contains("href="));
}

#[test]
fn failed_rule_bans_are_not_presented_as_successful_actions() {
    let mut case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    case.matched_rule_id = Some(42);
    case.status = "ban_failed".to_string();
    let text = action_log(&case);
    assert!(text.starts_with("<b>封禁失敗</b>"));
    assert!(text.contains("#42"));
    case.status = "auto_banned".to_string();
    assert!(action_log(&case).starts_with("<b>自動封禁</b>"));
    let text = group_notice(&case, "<b>已執行管理操作</b>", None, None);
    assert!(text.starts_with("<b>自動封禁</b>"));
}

#[tokio::test]
async fn error_diagnostics_redact_credentials_before_clipping() {
    let runtime = test_runtime().await;
    let mut config = runtime.config.clone();
    config.bot_token = "123:private-bot-token".to_string();
    config.hostctl_secret = Some("private-host-password".to_string());
    let text = format!(
        "https://api.telegram.org/bot{}/getMe\n{}\r{}",
        config.bot_token,
        config.hostctl_secret.as_ref().unwrap(),
        "error ".repeat(500)
    );
    let clean = diagnostic(&config, &text);
    assert!(!clean.contains(&config.bot_token));
    assert!(!clean.contains(config.hostctl_secret.as_ref().unwrap()));
    assert!(!clean.contains(['\r', '\n']));
    assert!(clean.encode_utf16().count() <= 1024);
    assert!(clean.contains("[redacted]"));
}

#[test]
fn health_failures_remain_visible_without_logging_every_probe() {
    let mut failures = HealthFailures::default();
    assert_eq!(failures.recovered(), None);
    let warnings: Vec<_> = (0..25).filter_map(|_| failures.failed()).collect();
    assert_eq!(warnings, vec![1, 10, 20]);
    assert_eq!(failures.recovered(), Some(25));
    assert_eq!(failures.recovered(), None);
    assert_eq!(failures.failed(), Some(1));
}
