use super::*;

pub(super) async fn moderate_edited_message(
    bot: Bot,
    runtime: Arc<Runtime>,
    message: Message,
) -> ResponseResult<()> {
    if (!message.chat.is_group() && !message.chat.is_supergroup())
        || runtime.is_group_banned(message.chat.id.0).await
    {
        return Ok(());
    }
    if runtime.config.test_group_id == Some(message.chat.id.0) {
        return score_only(&bot, &runtime, &message).await;
    }
    if check_captcha_and_act(&bot, &runtime, &message).await
        || check_guest_bot_and_act(&bot, &runtime, &message).await
        || check_project_ban_and_act(&bot, &runtime, &message).await
        || check_netban_and_act(&bot, &runtime, &message).await
        || check_reban_and_act(&bot, &runtime, &message).await
        || check_attachment_policy_and_act(&bot, &runtime, &message).await
    {
        return Ok(());
    }
    if ensure_bot_can_moderate(&bot, &runtime, message.chat.id).await? {
        auto_moderate(bot, runtime, message).await?;
    }
    Ok(())
}
