//! Owner confirmation cards for existing SPB logs. Only the bridge ledger is changed.
use super::*;
use rusqlite::OptionalExtension;

#[derive(Clone)]
pub(super) struct Request {
    pub id: String,
    pub case_id: String,
    pub user: i64,
    pub source_chat: i64,
    pub title: String,
    pub actor: i64,
    pub created: i64,
    pub status: String,
    pub review_state: String,
    pub review_chat: Option<i64>,
    pub review_message: Option<i32>,
}

pub(super) fn load(db: &Connection, id: &str) -> Result<Option<Request>> {
    Ok(db.query_row("SELECT id,case_id,user_id,source_chat,title,submitted_by,created_at,status,review_state,review_chat,review_message FROM watch_log_requests WHERE id=?1", [id], |r| Ok(Request {
        id:r.get(0)?,case_id:r.get(1)?,user:r.get(2)?,source_chat:r.get(3)?,title:r.get(4)?,actor:r.get(5)?,created:r.get(6)?,status:r.get(7)?,review_state:r.get(8)?,review_chat:r.get(9)?,review_message:r.get(10)?,
    })).optional()?)
}

pub(super) fn owner(db: &Connection) -> Result<i64> {
    let value: String =
        db.query_row("SELECT value FROM meta WHERE key='watch_owner'", [], |r| {
            r.get(0)
        })?;
    Ok(serde_json::from_str(&value)?)
}

pub(super) fn valid_case(source: &Connection, bridge: &Connection, row: &Request) -> Result<bool> {
    let valid: bool = source.query_row("SELECT EXISTS(SELECT 1 FROM cases c LEFT JOIN origin_ban_jobs j ON j.case_id=c.id
        WHERE c.id=?1 AND c.target_user_id=?2 AND c.chat_id=?3 AND c.netban_eligible=1
        AND (j.case_id IS NULL OR (j.ban_done=1 AND j.outcome_unknown=0))
        AND ((?4='訪客模式機器人封禁' AND c.action='guest_bot_ban' AND c.status='guest_bot_banned')
          OR (?4='自動封禁' AND c.action='auto_ban' AND c.status IN ('auto_banned','banned_delete_failed'))
          OR (?4='受理封禁' AND c.action='report_approved' AND c.status IN ('approved_and_banned','banned_delete_failed'))
          OR (?4='逕行封禁' AND c.action='spam_ban' AND c.status='force_approved')))" ,
        params![row.case_id,row.user,row.source_chat,row.title], |r| r.get(0))?;
    let bot = watch_spy::settings(bridge)?["bot_id"].as_i64();
    Ok(valid
        && row.user > 0
        && row.user < (1_i64 << 52)
        && Some(row.user) != bot
        && !watch_inbox::protected_in_spb(source, row.user)?
        && !watch_inbox::protected_in_bridge(bridge, row.user)?)
}

pub(super) fn decide(
    source: &Connection,
    bridge: &mut Connection,
    id: &str,
    actor: i64,
    chat: i64,
    message: i32,
    approve: bool,
) -> Result<&'static str> {
    let tx = bridge.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    anyhow::ensure!(actor == owner(&tx)?, "只有主持人可以確認轉發的 log");
    anyhow::ensure!(watch_spy::destination(&tx)? == chat, "請在指定報告群確認");
    let row = load(&tx, id)?.context("找不到這次確認請求")?;
    anyhow::ensure!(
        row.actor == actor
            && row.review_state == "sent"
            && row.review_chat == Some(chat)
            && row.review_message == Some(message),
        "請使用最新的原始確認卡"
    );
    if row.status != "pending" {
        return Ok("此請求已處理，未重複推送");
    }
    let now = Utc::now().timestamp();
    let (status, body) = if !approve {
        ("rejected", "已取消，這次轉發不會加入 hexDrill")
    } else if !(0..=86400).contains(&(now - row.created)) || !valid_case(source, &tx, &row)? {
        (
            "invalid",
            "案件已失效、受保護或確認卡已過期，未推送；請重新轉發 log",
        )
    } else {
        ("approved", "已確認加入 hexDrill +spb，等待推送及成功回覆")
    };
    tx.execute("UPDATE watch_log_requests SET status=?2,approved_by=?3,updated_at=?4 WHERE id=?1 AND status='pending'",
        params![id,status,if status=="approved" {Some(actor)} else {None},now])?;
    tx.commit()?;
    Ok(body)
}

fn card(case: &CaseRecord) -> String {
    notices::review_card(case,"WATCH · 加入 hexDrill？",
        "確認後加入 hexDrill +spb。取消則不因這次轉發推送。僅主持人可確認；不重新處置來源群組，也不訓練模型。",None)
}

pub(super) async fn callback(bot: &Bot, runtime: &Runtime, q: CallbackQuery) -> ResponseResult<()> {
    let data = q.data.as_deref().unwrap_or("");
    let parts: Vec<_> = data.split(':').collect();
    if parts.len() != 3 || !matches!(parts[1], "yes" | "no") || Uuid::parse_str(parts[2]).is_err() {
        bot.answer_callback_query(q.id)
            .text("無效的確認按鈕")
            .await?;
        return Ok(());
    }
    let Some(message) = q.message.as_ref() else {
        bot.answer_callback_query(q.id)
            .text("找不到原始確認卡")
            .await?;
        return Ok(());
    };
    let (id, actor, chat, mid, approve) = (
        parts[2].to_string(),
        q.from.id.0 as i64,
        message.chat().id.0,
        message.id().0,
        parts[1] == "yes",
    );
    let result = runtime
        .with_conn(move |conn| {
            let source = conn.transaction()?;
            let mut bridge = watch_inbox::private_db()?;
            let body = decide(&source, &mut bridge, &id, actor, chat, mid, approve)?;
            let row = load(&bridge, &id)?.context("missing confirmed request")?;
            Ok((body, row.user, row.case_id, row.status))
        })
        .await;
    match result {
        Ok((body, user, case, status)) => {
            bot.answer_callback_query(q.id).text(body).await?;
            let label = match status.as_str() {
                "approved" => "已確認，等待推送及 hexDrill 成功回覆",
                "rejected" => "已取消這次推送",
                _ => "案件已失效或確認卡已過期，未推送",
            };
            // The durable decision is already saved; a failed UI edit cannot approve twice.
            let _ = bot
                .edit_message_reply_markup(ChatId(chat), MessageId(mid))
                .reply_markup(InlineKeyboardMarkup::default())
                .await;
            let _ = bot
                .edit_message_text(
                    ChatId(chat),
                    MessageId(mid),
                    format!(
                        "WATCH · hexDrill 確認結果\n對象：{user}\n案例：#{}\n{label}",
                        case.replace('-', "").chars().take(8).collect::<String>()
                    ),
                )
                .await;
        }
        Err(error) => {
            log::warn!(
                "WATCH log confirmation: {}",
                notices::diagnostic(&runtime.config, &error.to_string())
            );
            bot.answer_callback_query(q.id)
                .text("未能確認：請由主持人在指定群組使用最新確認卡，或重新轉發 log")
                .await?;
        }
    }
    Ok(())
}

async fn tick(bot: &Bot, runtime: &Runtime) -> Result<()> {
    let bridge = watch_inbox::private_db()?;
    let exists: bool=bridge.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='watch_log_requests')",[],|r|r.get(0))?;
    if !exists {
        return Ok(());
    }
    let id: Option<String>=bridge.query_row("SELECT id FROM watch_log_requests WHERE status='pending' AND review_state='queued' ORDER BY created_at,id LIMIT 1",[],|r|r.get(0)).optional()?;
    let Some(id) = id else {
        return Ok(());
    };
    let chat = watch_spy::destination(&bridge)?;
    let row = load(&bridge, &id)?.context("missing log request")?;
    let item = row.clone();
    let valid = runtime
        .with_conn(move |conn| valid_case(conn, &watch_inbox::private_db()?, &item))
        .await?;
    if !valid
        || row.actor != owner(&bridge)?
        || !(0..=86400).contains(&(Utc::now().timestamp() - row.created))
    {
        bridge.execute("UPDATE watch_log_requests SET status='invalid',updated_at=?2 WHERE id=?1 AND status='pending'",params![id,Utc::now().timestamp()])?;
        return Ok(());
    }
    let case = runtime
        .load_case(&row.case_id)
        .await?
        .context("missing log case")?;
    let claimed=bridge.execute("UPDATE watch_log_requests SET review_state='sending',review_chat=?2,updated_at=?3 WHERE id=?1 AND status='pending' AND review_state='queued'",params![row.id,chat,Utc::now().timestamp()])?;
    if claimed == 0 {
        return Ok(());
    }
    let markup = InlineKeyboardMarkup::new(vec![vec![
        InlineKeyboardButton::callback("確認加入 +spb", format!("log:yes:{id}")),
        InlineKeyboardButton::callback("取消", format!("log:no:{id}")),
    ]]);
    let mut body = card(&case);
    body.push_str(&format!("\n封禁類型：{}", escape_html(&row.title)));
    if let Some(log) = case.log_message_id {
        body.push_str(&format!(
            "\n<a href=\"{}\">原始 log</a>",
            public_log_link(&runtime.config, log)
        ));
    }
    match bot
        .send_message(ChatId(chat), body)
        .parse_mode(ParseMode::Html)
        .reply_markup(markup)
        .await
    {
        Ok(message) => {
            bridge.execute("UPDATE watch_log_requests SET review_state='sent',review_message=?2,updated_at=?3 WHERE id=?1 AND review_state='sending'",params![id,message.id.0,Utc::now().timestamp()])?;
        }
        Err(_) => {
            bridge.execute("UPDATE watch_log_requests SET review_state='uncertain',review_error='send_unconfirmed',updated_at=?2 WHERE id=?1 AND review_state='sending'",params![id,Utc::now().timestamp()])?;
        }
    }
    Ok(())
}

pub(super) fn spawn(bot: Bot, runtime: Arc<Runtime>) {
    if env::var("WATCH_SPY_ENABLED").as_deref() != Ok("1") {
        return;
    }
    tokio::spawn(async move {
        // Never blindly resend a card after an interrupted send.
        if let Ok(db) = watch_inbox::private_db() {
            let _=db.execute("UPDATE watch_log_requests SET review_state='uncertain',review_error='restart_during_send' WHERE review_state='sending'",[]);
        }
        loop {
            if let Err(error) = tick(&bot, &runtime).await {
                log::warn!(
                    "WATCH log card: {}",
                    notices::diagnostic(&runtime.config, &error.to_string())
                );
            }
            sleep(Duration::from_secs(3)).await;
        }
    });
}
