//! Telegram sender chats use signed chat IDs, never user IDs. Keep the
//! distinction through reporting, queued bans, network delivery and reversal.
use super::*;

pub(super) fn reply_context(message: &Message) -> Result<(i64, String, i32, String), &'static str> {
    let reply = real_reply(message).ok_or("請回覆一條疑似 spam 的訊息。")?;
    let (id, name) = if let Some(chat) = &reply.sender_chat {
        if !chat.is_channel() || chat.id == message.chat.id {
            return Err("此訊息由匿名管理員以群組身分發送，無法辨識個別用戶。");
        }
        if reply.is_automatic_forward() {
            return Err("此訊息是關聯頻道的自動轉貼，不能作為封禁舉報。");
        }
        let title = chat.title().unwrap_or("頻道");
        let name = match chat.username() {
            Some(username) => format!("{title} (@{username})"),
            None => title.to_string(),
        };
        (chat.id.0, name)
    } else {
        let user = reply.from.as_ref().ok_or("無法辨識此訊息的發送者。")?;
        if is_platform_pseudo_user(user.id.0 as i64) {
            return Err("此訊息只有 Telegram 代發帳號，無法辨識實際發送者。");
        }
        // A normal forward still belongs to its immediate sender. Its
        // forward_origin is evidence, not authority to ban that origin.
        (user.id.0 as i64, short_user(user))
    };
    Ok((id, name, reply.id.0, extract_full_text(reply)))
}

pub(super) async fn protected(bot: &Bot, chat: ChatId, target: i64) -> ResponseResult<bool> {
    if target < 0 {
        if chat.0 == target {
            return Ok(true);
        }
        // Preserve the exemption for a group's own linked channel, including
        // when a queued network ban is delivered after the link changes.
        Ok(bot.get_chat(chat).await?.linked_chat_id() == Some(target))
    } else {
        Ok(bot
            .get_chat_member(chat, UserId(target as u64))
            .await?
            .kind
            .is_privileged())
    }
}

pub(super) async fn ban(bot: &Bot, chat: ChatId, target: i64) -> ResponseResult<()> {
    if target < 0 {
        bot.ban_chat_sender_chat(chat, ChatId(target)).await?;
    } else {
        bot.ban_chat_member(chat, UserId(target as u64)).await?;
    }
    Ok(())
}

pub(super) async fn unban(
    bot: &Bot,
    chat: ChatId,
    target: i64,
    only_if_banned: bool,
) -> ResponseResult<()> {
    if target < 0 {
        bot.unban_chat_sender_chat(chat, ChatId(target)).await?;
    } else {
        bot.unban_chat_member(chat, UserId(target as u64))
            .only_if_banned(only_if_banned)
            .await?;
    }
    Ok(())
}
