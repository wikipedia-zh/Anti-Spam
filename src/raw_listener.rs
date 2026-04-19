use regex::Regex;
use reqwest::Client;
use serde_json::Value;
use teloxide::{prelude::Requester, stop::{mk_stop_token, StopToken}, types::{AllowedUpdate, ChatId, MessageId, Update, UserId}};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

pub struct RawJsonPolling {
    bot: teloxide::Bot,
    client: Client,
    token: String,
    offset: i64,
    stop_token: StopToken,
    rules: Vec<Regex>,
}

impl RawJsonPolling {
    pub fn new(bot: teloxide::Bot, token: String, rules: Vec<Regex>) -> Self {
        let (stop_token, _flag) = mk_stop_token();
        Self { bot, client: Client::new(), token, offset: 0, stop_token, rules }
    }

    fn updates_url(&self) -> String {
        format!("https://api.telegram.org/bot{}/getUpdates", self.token)
    }
}

impl teloxide::update_listeners::UpdateListener for RawJsonPolling {
    type Err = reqwest::Error;

    fn stop_token(&mut self) -> StopToken {
        self.stop_token.clone()
    }

    fn hint_allowed_updates(&mut self, hint: &mut dyn Iterator<Item = AllowedUpdate>) {
        let _ = hint;
    }
}

impl<'a> teloxide::update_listeners::AsUpdateStream<'a> for RawJsonPolling {
    type StreamErr = reqwest::Error;
    type Stream = ReceiverStream<Result<Update, reqwest::Error>>;

    fn as_stream(&'a mut self) -> Self::Stream {
        let (tx, rx) = mpsc::channel(64);
        let bot = self.bot.clone();
        let client = self.client.clone();
        let url = self.updates_url();
        let mut offset = self.offset;
        let rules = self.rules.clone();
        tokio::spawn(async move {
            loop {
                let response = match client
                    .get(&url)
                    .query(&[("offset", offset.to_string()), ("timeout", "30".to_string())])
                    .send()
                    .await
                {
                    Ok(resp) => resp,
                    Err(err) => {
                        let _ = tx.send(Err(err)).await;
                        continue;
                    }
                };

                let raw = match response.text().await {
                    Ok(text) => text,
                    Err(err) => {
                        let _ = tx.send(Err(err)).await;
                        continue;
                    }
                };

                let value: Value = match serde_json::from_str(&raw) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let Some(updates) = value.get("result").and_then(|r| r.as_array()) else {
                    continue;
                };

                for upd in updates {
                    if let Some(id) = upd.get("update_id").and_then(|v| v.as_i64()) {
                        offset = id + 1;
                    }

                    let upd_str = upd.to_string();
                    let is_spam = rules.iter().any(|rule| rule.is_match(&upd_str));

                    if is_spam {
                        if let Some(message) = upd.get("message") {
                            if let (Some(chat_id), Some(message_id), Some(user_id)) = (
                                message.get("chat").and_then(|c| c.get("id")).and_then(|v| v.as_i64()),
                                message.get("message_id").and_then(|v| v.as_i64()),
                                message.get("from").and_then(|f| f.get("id")).and_then(|v| v.as_i64()),
                            ) {
                                if let Ok(message_id) = i32::try_from(message_id) {
                                    let _ = bot.delete_message(ChatId(chat_id), MessageId(message_id)).await;
                                }
                                let _ = bot.ban_chat_member(ChatId(chat_id), UserId(user_id as u64)).await;
                                log::info!("Raw Interceptor: Banned user {} and deleted message {} due to Regex match.", user_id, message_id);
                            }
                        }
                        continue;
                    }

                    if let Ok(update) = serde_json::from_value::<Update>(upd.clone()) {
                        let _ = tx.send(Ok(update)).await;
                    }
                }
            }
        });

        ReceiverStream::new(rx)
    }
}
