use super::*;
use rusqlite::OptionalExtension;

pub(super) const MARKER: &str = "SEELE_DM";

pub(super) fn is_case(case: &CaseRecord) -> bool {
    case.chat_id > 0
        && case.chat_id == case.target_user_id
        && case.matched_rule_pattern.as_deref() == Some(MARKER)
}

#[derive(Deserialize)]
pub(super) struct NetbanRequest {
    pub request_id: String,
    pub id: i64,
    pub operator_id: i64,
    pub host_chat_id: i64,
    pub host_message_id: i32,
    pub source_message_id: i32,
}

pub(super) fn admit(
    conn: &mut Connection,
    request: &NetbanRequest,
    host_chat: i64,
    test_group: Option<i64>,
) -> Result<(String, bool)> {
    let id = Uuid::parse_str(&request.request_id)?.to_string();
    anyhow::ensure!(
        request.id > 0
            && request.id < (1_i64 << 52)
            && request.host_chat_id == host_chat
            && host_chat < 0
            && request.source_message_id > 0
            && request.host_message_id > 0,
        "無效的 SEELE 請求"
    );
    let tx = conn.transaction()?;
    let authorized: bool = is_host(request.operator_id) || tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM maintainers WHERE user_id=?1) OR EXISTS(SELECT 1 FROM reviewers WHERE user_id=?1)",
        [request.operator_id], |row| row.get(0))?;
    anyhow::ensure!(authorized, "只限 SPB 維護組或審核員加入聯防黑名單");
    let previous: Option<(i64, String, String)> = tx
        .query_row(
            "SELECT target_user_id,COALESCE(matched_rule_pattern,''),status FROM cases WHERE id=?1",
            [&id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    if let Some((user, marker, status)) = previous {
        anyhow::ensure!(user == request.id && marker == MARKER, "請求編號與案件不符");
        anyhow::ensure!(
            status == "approved_and_banned",
            "此案件已撤銷或正在撤銷，不會重複封禁"
        );
        anyhow::ensure!(
            !watch_inbox::protected_in_spb(&tx, request.id)?,
            "此帳號目前受保護"
        );
        return Ok((id, false));
    }
    anyhow::ensure!(
        !watch_inbox::protected_in_spb(&tx, request.id)?,
        "此帳號目前受保護，不能加入聯防黑名單"
    );
    let evidence = format!(
        "SEELE 私訊垃圾廣告，管理員核准。證據保留於管理群；訊息 {}。",
        request.host_message_id
    );
    tx.execute("INSERT INTO cases(id,action,chat_id,target_user_id,target_name,actor_user_id,actor_name,source_message_id,evidence_text,matched_rule_pattern,status,created_at)
        VALUES (?1,'report_approved',?2,?2,?3,?4,'SEELE 管理員',?5,?6,'SEELE_DM','approved_and_banned',?7)",
        params![id,request.id,request.id.to_string(),request.operator_id,request.source_message_id,evidence,Utc::now().to_rfc3339()])?;
    network_delivery::enqueue_network_ban(&tx, &id, test_group)?;
    tx.commit()?;
    Ok((id, true))
}

pub(super) async fn handle_netban(
    bot: &Bot,
    runtime: &Runtime,
    chat: i64,
    data: serde_json::Value,
) {
    let request_id = data["request_id"].as_str().unwrap_or_default().to_owned();
    let user = data["id"].as_i64().unwrap_or(0);
    let outcome = async {
        let request: NetbanRequest = serde_json::from_value(data)?;
        let host: i64 = env::var("SEELE_HOST_CHAT_ID")?.parse()?;
        let test_group = runtime.config.test_group_id;
        anyhow::ensure!(
            request.id != bot.get_me().await?.id.0 as i64,
            "不能封禁服務帳號"
        );
        let _guard = runtime.user_action_guard(request.id).await;
        runtime
            .with_conn(move |conn| admit(conn, &request, host, test_group))
            .await
    }
    .await;
    let (success, case_id, message) = match outcome {
        Ok((id, fresh)) => {
            if fresh {
                if let Ok(Some(mut case)) = runtime.load_case(&id).await {
                    if let Ok(mid) = log_action(bot, runtime, &case).await {
                        case.log_message_id = Some(mid);
                        let _ = runtime.persist_case(&case).await;
                    }
                    broadcast_ban_status(bot, runtime, user, true).await;
                }
            }
            (
                true,
                case_ids::display(&id),
                "已加入 SPB netban 黑名單；各群組封禁由佇列處理。".to_string(),
            )
        }
        Err(err) => (false, String::new(), err.to_string()),
    };
    send_exchange_message(
        bot,
        chat,
        "result",
        "netban",
        serde_json::json!({
            "id":user,"request_id":request_id,"success":success,"case_id":case_id,"message":message,
        }),
    )
    .await;
}

pub(super) async fn check_unban(bot: &Bot, runtime: &Runtime, chat: i64, data: serde_json::Value) {
    let Some(user) = data["id"].as_i64() else {
        return;
    };
    let Some(ids) = data["case_ids"]
        .as_array()
        .filter(|v| !v.is_empty() && v.len() <= 500)
    else {
        return;
    };
    let Some(request) = data["request_id"].as_str().filter(|s| !s.is_empty()) else {
        return;
    };
    for id in ids {
        let Some(id) = id.as_str() else { return };
        match runtime.load_case(id).await {
            Ok(Some(case))
                if case.target_user_id == user
                    && case.status == "reversed"
                    && case.action == ActionKind::Unbanned => {}
            _ => return,
        }
    }
    match runtime.find_active_bans_for_user(user).await {
        Ok(cases) if cases.is_empty() => {}
        _ => return,
    }
    send_exchange_message(
        bot,
        chat,
        "result",
        "unban",
        serde_json::json!({
            "id":user,"request_id":request,"success":true,"verified":true,
            "case_ids":ids,"message":"SPB 已確認本次申訴涉及的封禁全部解除。"
        }),
    )
    .await;
}
