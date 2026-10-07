use super::*;

pub(super) async fn handle(
    bot: &Bot,
    runtime: &Runtime,
    message: &Message,
    case_id: &str,
) -> ResponseResult<()> {
    let Some(from) = message.from.as_ref() else {
        return Ok(());
    };
    if !runtime.is_maintainer(from.id.0 as i64).await {
        return handle_permission_denied(
            bot,
            runtime,
            message,
            from,
            "只有項目維護組可以查詢工作佇列。",
        )
        .await;
    }
    if !message.chat.is_private() {
        bot.send_message(message.chat.id, "請在私聊中使用 /queue。")
            .await?;
        return Ok(());
    }
    let filter = (!case_id.is_empty()).then_some(case_id);
    let text = runtime.queue_status(filter).await.unwrap_or_else(|err| {
        log::warn!("queue status: {err}");
        "無法讀取工作佇列，請稍後重試。".into()
    });
    bot.send_message(message.chat.id, text)
        .parse_mode(ParseMode::Html)
        .await?;
    Ok(())
}

// The same completion predicates used by the workers, including notices
// whose case has changed since the previous successful delivery.
const WORK: &str = "
    SELECT '封禁' AS kind,j.case_id,c.chat_id,j.attempts,j.next_attempt_at,j.last_error
        FROM origin_ban_jobs j JOIN cases c ON c.id=j.case_id WHERE j.state='pending'
    UNION ALL SELECT '跨群封禁',case_id,chat_id,attempts,next_attempt_at,last_error
        FROM network_deliveries WHERE state='pending'
    UNION ALL SELECT '禁言／踢人',j.case_id,c.chat_id,j.attempts,j.next_attempt_at,j.last_error
        FROM restriction_jobs j JOIN cases c ON c.id=j.case_id WHERE j.state='pending'
    UNION ALL SELECT '解封',j.case_id,c.chat_id,j.attempts,j.next_attempt_at,j.last_error
        FROM reversal_retries j JOIN cases c ON c.id=j.case_id WHERE c.status='reversal_pending'
    UNION ALL SELECT '入群驗證',NULL,chat_id,attempts,next_attempt_at,last_error FROM captcha_jobs
    UNION ALL SELECT '審核通知',u.case_id,u.chat_id,u.attempts,u.next_attempt_at,u.last_error
        FROM review_updates u JOIN cases c ON c.id=u.case_id LEFT JOIN origin_ban_jobs j ON j.case_id=c.id
        WHERE u.review_status!=c.status||':'||COALESCE(j.state,'') OR u.confirmation_status!=c.status||':'||COALESCE(j.state,'')";

struct Item {
    kind: String,
    case_id: Option<String>,
    chat: i64,
    attempts: i64,
    next: i64,
    error: Option<String>,
}

impl Runtime {
    pub(super) async fn queue_status(&self, case_id: Option<&str>) -> Result<String> {
        let case_id = case_id.map(str::to_string);
        let now = Utc::now().timestamp();
        let (groups,items,cooldown,reports)=self.with_conn(move |conn| {
            let tx=conn.transaction()?;
            let cooldown:i64=tx.query_row("SELECT not_before FROM telegram_retry_state WHERE id=1",[],|r|r.get(0))?;
            let groups={
                let mut stmt=tx.prepare(&format!("SELECT kind,COUNT(*),SUM(last_error IS NOT NULL),SUM(next_attempt_at<=?2 AND ?3<=?2) FROM ({WORK}) WHERE (?1 IS NULL OR case_id=?1) GROUP BY kind ORDER BY kind"))?;
                let rows=stmt.query_map(params![case_id,now,cooldown],|r|Ok((r.get::<_,String>(0)?,r.get::<_,i64>(1)?,r.get::<_,i64>(2)?,r.get::<_,i64>(3)?)))?;
                rows.collect::<rusqlite::Result<Vec<_>>>()?
            };
            let items={
                let mut stmt=tx.prepare(&format!("SELECT kind,case_id,chat_id,attempts,next_attempt_at,last_error FROM ({WORK}) WHERE (?1 IS NULL OR case_id=?1) ORDER BY (last_error IS NOT NULL) DESC,attempts DESC,next_attempt_at,case_id LIMIT 5"))?;
                let rows=stmt.query_map(params![case_id],|r|Ok(Item{kind:r.get(0)?,case_id:r.get(1)?,chat:r.get(2)?,attempts:r.get(3)?,next:r.get(4)?,error:r.get(5)?}))?;
                rows.collect::<rusqlite::Result<Vec<_>>>()?
            };
            let reports:i64=tx.query_row("SELECT COUNT(*) FROM cases WHERE action='pending_report' AND status='pending_review' AND (?1 IS NULL OR id=?1)",params![case_id],|r|r.get(0))?;
            tx.commit()?;Ok((groups,items,cooldown,reports))
        }).await?;
        let total: i64 = groups.iter().map(|(_, n, _, _)| n).sum();
        let mut text = format!("<b>工作佇列</b>\n未完成：{total} · 待人工審核：{reports}");
        if cooldown > now {
            text.push_str(&format!(
                "\nTelegram 限流，約 {} 秒後恢復。",
                cooldown - now
            ));
        }
        for (kind, count, failed, due) in groups {
            text.push_str(&format!(
                "\n{kind}：{count}（待執行 {due} · 有錯誤 {failed}）"
            ));
        }
        if total == 0 {
            text.push_str("\n沒有未完成的背景工作。");
        }
        for item in items {
            text.push_str(&format!(
                "\n\n<b>{}</b> · 已嘗試 {} 次\n群組：<code>{}</code>",
                item.kind, item.attempts, item.chat
            ));
            if let Some(id) = item.case_id {
                text.push_str(&format!(
                    "\n案例：<code>{}</code>",
                    escape_html(&notices::preview(&id, 64))
                ));
            }
            let wait = item.next.max(cooldown).saturating_sub(now).max(0);
            text.push_str(&if wait == 0 {
                "\n下一次：待執行".to_string()
            } else {
                format!("\n下一次：約 {wait} 秒後")
            });
            if let Some(error) = item.error {
                text.push_str(&format!(
                    "\n錯誤：{}",
                    escape_html(&notices::preview(
                        &notices::diagnostic(&self.config, &error),
                        180
                    ))
                ));
            }
        }
        if total > 5 {
            text.push_str("\n\n僅列出 5 項；用 /queue 案例ID 查詢單一案例。");
        }
        Ok(text)
    }
}
