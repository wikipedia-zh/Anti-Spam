use super::*;

// Telegram counts text after parsing HTML. Clip before escaping so entities
// and surrogate pairs stay intact; keep the full evidence in the case record.
pub(super) fn preview(text: &str, limit: usize) -> String {
    if text.encode_utf16().count() <= limit {
        return text.to_string();
    }
    let mut used = 0;
    let mut out = String::new();
    for ch in text.chars() {
        if used + ch.len_utf16() > limit.saturating_sub(1) {
            break;
        }
        out.push(ch);
        used += ch.len_utf16();
    }
    out.push('…');
    out
}

fn field(text: &str, limit: usize) -> String {
    escape_html(&preview(text, limit))
}

fn evidence(case: &CaseRecord) -> String {
    if case.evidence_text.trim().is_empty() {
        return String::new();
    }
    let short = preview(&case.evidence_text, 2400);
    let note = if short != case.evidence_text {
        "\n（證據過長，完整內容已保存在案例記錄。）"
    } else {
        ""
    };
    format!("\n<blockquote>{}</blockquote>{note}", escape_html(&short))
}

fn identity(case: &CaseRecord) -> String {
    format!(
        "\n<b>對象</b>: <code>{}</code> {}\n<b>群組</b>: <code>{}</code>\n<b>案例</b>: <code>{}</code>",
        case.target_user_id, field(&case.target_name, 256), case.chat_id, field(&case.id, 64),
    )
}

fn reason(case: &CaseRecord, link: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(id) = case.matched_rule_id {
        out.push_str(&format!("\n<b>規則</b>: #{id}"));
    }
    if let Some(text) = case
        .matched_rule_pattern
        .as_deref()
        .filter(|s| !s.trim().is_empty() && *s != "-")
    {
        out.push_str(&format!(
            "\n<b>原因</b>: {}",
            format_public_reason(&preview(text, 384), link)
        ));
    }
    out
}

pub(super) fn action_log(case: &CaseRecord) -> String {
    let mut out = format!("<b>{}</b>{}", chinese_case_action(case), identity(case));
    if let Some(actor) = case.actor_user_id {
        out.push_str(&format!("\n<b>處理者</b>: <code>{actor}</code>"));
    }
    if let Some(score) = case.model_score {
        out.push_str(&format!("\n<b>分數</b>: {score:.4}"));
    }
    out.push_str(&reason(case, None));
    out.push_str(&format!("\n{}", utc8_display(case.created_at)));
    out.push_str(&evidence(case));
    out
}

pub(super) fn group_notice(
    case: &CaseRecord,
    header: &str,
    log_link: Option<&str>,
    reason_link: Option<&str>,
) -> String {
    let title = match header {
        "<b>已執行管理操作</b>" => format!("<b>{}</b>", chinese_case_action(case)),
        "<b>警告達門檻，已自動處置</b>" => {
            format!("<b>警告達門檻：{}</b>", chinese_case_action(case))
        }
        _ => header.to_string(),
    };
    let mut out = format!("{title}\n<b>對象</b>: <code>{}</code>", case.target_user_id);
    out.push_str(&reason(case, reason_link));
    out.push_str(&format!(
        "\n<b>案例</b>: <code>{}</code>",
        field(&case.id, 64)
    ));
    if let Some(link) = log_link {
        out.push_str(&format!("\n<a href=\"{link}\">查看日誌</a>"));
    }
    out
}

pub(super) fn review_card(
    case: &CaseRecord,
    title: &str,
    note: &str,
    reviewer: Option<i64>,
) -> String {
    let mut out = format!("<b>{title}</b>{}", identity(case));
    if let Some(id) = case.actor_user_id {
        out.push_str(&format!(
            "\n<b>發起人</b>: <code>{id}</code> {}",
            field(case.actor_name.as_deref().unwrap_or(""), 128)
        ));
    }
    if let Some(id) = reviewer {
        out.push_str(&format!("\n<b>審核員</b>: <code>{id}</code>"));
    }
    out.push_str(&evidence(case));
    if !note.is_empty() {
        out.push_str(&format!("\n\n{note}"));
    }
    out
}

pub(super) fn case_lookup(case: &CaseRecord, link: &str, reason_link: &str) -> String {
    let mut out = action_log(case);
    // Keep the stored status available for troubleshooting without giving it
    // a second prominent heading beside the result.
    out.push_str(&format!("\n<code>{}</code>", field(&case.status, 64)));
    if link != "-" {
        out.push_str(&format!("\n<a href=\"{link}\">查看日誌</a>"));
    }
    if reason_link != "-" && reason_link != link {
        out.push_str(&format!(" · <a href=\"{reason_link}\">原因說明</a>"));
    }
    out
}

pub(super) fn diagnostic(config: &Config, text: &str) -> String {
    let mut clean = text.to_string();
    for secret in [
        Some(config.bot_token.as_str()),
        config.hostctl_secret.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        if !secret.is_empty() {
            clean = clean.replace(secret, "[redacted]");
        }
    }
    preview(&clean.replace(['\r', '\n'], " "), 1024)
}

pub(super) fn error_stage(stage: &str) -> &str {
    match stage {
        "ban" => "封禁",
        "delete_message" => "刪除訊息",
        "training_review" => "訓練審核",
        "train_spam" | "train_ham" => "更新模型",
        "store_case" => "儲存案例",
        "log_action" => "發送日誌",
        _ => "處理案例",
    }
}

#[derive(Default)]
pub(super) struct HealthFailures(u64);

impl HealthFailures {
    pub(super) fn failed(&mut self) -> Option<u64> {
        self.0 = self.0.saturating_add(1);
        // First failure, then once per five minutes at the 30-second probe interval.
        (self.0 == 1 || self.0.is_multiple_of(10)).then_some(self.0)
    }

    pub(super) fn recovered(&mut self) -> Option<u64> {
        let count = std::mem::take(&mut self.0);
        (count > 0).then_some(count)
    }
}
