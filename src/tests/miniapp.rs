use super::*;
use crate::miniapp::{validate_init_data, Api, Service, Settings};
use hmac::{Hmac, Mac};
use sha2::Sha256;

fn signed(token: &str, user_id: i64, launch: &str, date: i64) -> String {
    let user = format!("{{\"id\":{user_id}}}");
    let check = format!("auth_date={date}\nstart_param={launch}\nuser={user}");
    let mut secret = Hmac::<Sha256>::new_from_slice(b"WebAppData").unwrap();
    secret.update(token.as_bytes());
    let mut mac = Hmac::<Sha256>::new_from_slice(&secret.finalize().into_bytes()).unwrap();
    mac.update(check.as_bytes());
    let hash = format!("{:x}", mac.finalize().into_bytes());
    url::form_urlencoded::Serializer::new(String::new())
        .append_pair("user", &user)
        .append_pair("auth_date", &date.to_string())
        .append_pair("start_param", launch)
        .append_pair("hash", &hash)
        .finish()
}

#[test]
fn telegram_signature_covers_every_field_and_rejects_replays_and_tampering() {
    // Independently calculated with .NET HMACSHA256, including Telegram's
    // optional signature field in the bot-token validation string.
    let raw = "user=%7B%22id%22%3A42%7D&auth_date=1000&start_param=0123456789abcdef0123456789abcdef&signature=test-signature&hash=d6ed53d10daabafd0cfa2a9f43eda0ac4a5aec979c77daa78d7fc2382b40c5e5";
    assert_eq!(
        validate_init_data(raw, "123:test-token", 1000)
            .unwrap()
            .user_id,
        42
    );
    assert!(validate_init_data(raw, "other-token", 1000).is_err());
    assert!(validate_init_data(raw, "123:test-token", 1301).is_err());
    assert!(validate_init_data(raw, "123:test-token", 969).is_err());
    assert!(validate_init_data(&raw.replace("%3A42", "%3A43"), "123:test-token", 1000).is_err());
    assert!(validate_init_data(&format!("{raw}&auth_date=1000"), "123:test-token", 1000).is_err());
    assert!(validate_init_data(
        &raw.replace("signature=test-signature&", ""),
        "123:test-token",
        1000
    )
    .is_err());
}

struct TestApi {
    runtime: Arc<Runtime>,
    service: Arc<Service>,
    telegram: TelegramStub,
    url: String,
    client: reqwest::Client,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for TestApi {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl TestApi {
    async fn new() -> Self {
        let runtime = Arc::new(test_runtime().await);
        runtime.me_id.set(UserId(999)).unwrap();
        let telegram = TelegramStub::with_members(vec![200, HOST_ID, 999], vec![], true);
        let service = Arc::new(Service::new(Settings {
            origin: "https://spb.example".into(),
            proxy_key: "test-proxy-key-with-at-least-32-characters".into(),
            launch_url: Url::parse("https://t.me/testbot").unwrap(),
            listen: "127.0.0.1:0".parse().unwrap(),
        }));
        assert!(runtime.miniapp.set(service.clone()).is_ok());
        let app = crate::miniapp::router(Api {
            bot: telegram.bot.clone(),
            runtime: runtime.clone(),
            service: service.clone(),
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            runtime,
            service,
            telegram,
            url,
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
            task,
        }
    }
    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        self.client
            .request(method, format!("{}{path}", self.url))
            .header("origin", "https://spb.example")
            .header(
                "x-spb-proxy-key",
                "test-proxy-key-with-at-least-32-characters",
            )
            .header("content-type", "application/json")
    }
    async fn init(&self, user: i64, chat: i64) -> String {
        let link = self.service.launch_link(user, chat).await.unwrap();
        let launch = link
            .query_pairs()
            .find(|(key, _)| key == "startapp")
            .unwrap()
            .1
            .into_owned();
        signed(
            &self.runtime.config.bot_token,
            user,
            &launch,
            Utc::now().timestamp(),
        )
    }
    async fn login_raw(&self, raw: &str) -> reqwest::Response {
        self.request(reqwest::Method::POST, "/api/miniapp/session")
            .body(serde_json::json!({"init_data":raw}).to_string())
            .send()
            .await
            .unwrap()
    }
    async fn login(&self, user: i64, chat: i64) -> String {
        let response = self.login_raw(&self.init(user, chat).await).await;
        let status = response.status();
        let body = response.text().await.unwrap();
        assert_eq!(status, 200, "{body}");
        serde_json::from_str::<serde_json::Value>(&body).unwrap()["token"]
            .as_str()
            .unwrap()
            .to_string()
    }
    async fn save(
        &self,
        token: &str,
        revision: i64,
        changes: serde_json::Value,
    ) -> reqwest::Response {
        self.request(reqwest::Method::PATCH,"/api/groups/current/settings").bearer_auth(token).body(serde_json::json!({"request_id":Uuid::new_v4().to_string(),"expected_revision":revision,"changes":changes}).to_string()).send().await.unwrap()
    }
}

#[tokio::test]
async fn group_admin_can_edit_ot_text_but_cannot_save_after_revocation() {
    let api = TestApi::new().await;
    let token = api.login(200, -100).await;
    let response = api
        .save(
            &token,
            0,
            serde_json::json!({"ot_template":"{user} 請留意群組主題。{count}"}),
        )
        .await;
    assert_eq!(response.status(), 200);
    let saved: serde_json::Value = serde_json::from_str(&response.text().await.unwrap()).unwrap();
    assert_eq!(saved["ot_template"], "{user} 請留意群組主題。{count}");
    let revision = saved["revision"].as_i64().unwrap();
    let invalid = api
        .save(&token, revision, serde_json::json!({"ot_template":" "}))
        .await;
    assert_eq!(invalid.status(), 400);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&invalid.text().await.unwrap()).unwrap()["error"],
        "invalid_template"
    );
    api.telegram.members.lock().unwrap().insert(
        (-100, 200),
        serde_json::json!({"status":"member","user":{"id":200,"is_bot":false,"first_name":"Test"}}),
    );
    assert_eq!(
        api.save(&token, revision, serde_json::json!({"ot_template":null}))
            .await
            .status(),
        403
    );
    assert!(api
        .runtime
        .get_warn_settings(-100)
        .await
        .unwrap()
        .ot_template
        .is_some());
}

#[tokio::test]
async fn only_the_bound_user_can_redeem_a_launch_and_only_once() {
    let api = TestApi::new().await;
    let raw = api.init(200, -100).await;
    let fields: HashMap<_, _> = url::form_urlencoded::parse(raw.as_bytes())
        .into_owned()
        .collect();
    let stranger = signed(
        &api.runtime.config.bot_token,
        201,
        &fields["start_param"],
        Utc::now().timestamp(),
    );
    assert_eq!(api.login_raw(&stranger).await.status(), 401);
    let (one, two) = tokio::join!(api.login_raw(&raw), api.login_raw(&raw));
    let mut statuses = [one.status().as_u16(), two.status().as_u16()];
    statuses.sort();
    assert_eq!(statuses, [200, 401]);
}

#[tokio::test]
async fn saves_use_the_session_group_and_recheck_revoked_permissions() {
    let api = TestApi::new().await;
    let token = api.login(200, -100).await;
    assert_eq!(
        api.save(&token, 0, serde_json::json!({"netban":true}))
            .await
            .status(),
        200
    );
    assert!(api.runtime.get_group_modules(-100).await.unwrap().netban);
    assert!(!api.runtime.get_group_modules(-200).await.unwrap().netban);
    assert_eq!(
        api.save(&token, 0, serde_json::json!({"captcha":true}))
            .await
            .status(),
        409
    );
    let body = serde_json::json!({"chat_id":-200,"request_id":Uuid::new_v4().to_string(),"expected_revision":1,"changes":{"captcha":true}});
    assert_eq!(
        api.request(reqwest::Method::PATCH, "/api/groups/current/settings")
            .bearer_auth(&token)
            .body(body.to_string())
            .send()
            .await
            .unwrap()
            .status(),
        400
    );
    api.telegram.members.lock().unwrap().insert((-100,200),serde_json::json!({"status":"member","user":{"id":200,"is_bot":false,"first_name":"Member"}}));
    assert_eq!(
        api.save(&token, 1, serde_json::json!({"captcha":true}))
            .await
            .status(),
        403
    );
    assert!(!api.runtime.get_group_modules(-100).await.unwrap().captcha);
}

#[tokio::test]
async fn threshold_privileges_and_hidden_modules_match_existing_commands() {
    let api = TestApi::new().await;
    let token = api.login(200, -100).await;
    assert_eq!(
        api.save(&token, 0, serde_json::json!({"threshold_override":0.9}))
            .await
            .status(),
        403
    );
    assert_eq!(
        api.save(&token, 0, serde_json::json!({"warn-pol":true}))
            .await
            .status(),
        400
    );
    let token = api.login(HOST_ID, -100).await;
    assert_eq!(
        api.save(&token, 0, serde_json::json!({"threshold_override":0.93}))
            .await
            .status(),
        200
    );
    let response = api
        .request(reqwest::Method::GET, "/api/groups/current/settings")
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(response.headers()["cache-control"], "no-store");
    let body: serde_json::Value = serde_json::from_str(&response.text().await.unwrap()).unwrap();
    assert_eq!(body["can_edit_threshold"], true);
    assert_eq!(body["settings"]["threshold_override"], 0.93);
    assert!(body["settings"]["modules"].get("warn-pol").is_none());
}

#[tokio::test]
async fn untrusted_origins_proxy_keys_and_missing_sessions_cannot_read_settings() {
    let api = TestApi::new().await;
    let token = api.login(200, -100).await;
    for request in [
        api.request(reqwest::Method::GET, "/api/groups/current/settings")
            .header("origin", "https://other.example")
            .bearer_auth(&token),
        api.request(reqwest::Method::GET, "/api/groups/current/settings")
            .header("x-spb-proxy-key", "wrong")
            .bearer_auth(&token),
    ] {
        assert_eq!(request.send().await.unwrap().status(), 403);
    }
    assert_eq!(
        api.request(reqwest::Method::GET, "/api/groups/current/settings")
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
}

#[tokio::test]
async fn settings_command_uses_a_group_link_bound_to_the_requesting_admin() {
    let api = TestApi::new().await;
    assert!(matches!(
        parse_command("/settings@testbot"),
        ModerationCommand::Settings
    ));
    crate::miniapp::launch(
        &api.telegram.bot,
        &api.runtime,
        &spam_ban_message(200, "/settings", None),
    )
    .await
    .unwrap();
    let calls = api.telegram.requests.lock().unwrap().clone();
    let (_, args) = calls
        .iter()
        .find(|(m, a)| m == "sendmessage" && a.get("reply_markup").is_some())
        .unwrap();
    assert!(args["reply_markup"]["inline_keyboard"][0][0]
        .get("web_app")
        .is_none());
    let link = args["reply_markup"]["inline_keyboard"][0][0]["url"]
        .as_str()
        .unwrap();
    assert!(link.starts_with("https://t.me/testbot?startapp="));
}
