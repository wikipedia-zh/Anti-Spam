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
async fn host_sessions_are_separate_from_group_admin_and_maintainer_sessions() {
    let api = TestApi::new().await;
    api.runtime
        .set_maintainer(200, true, Some(HOST_ID))
        .await
        .unwrap();
    assert!(api.service.host_link(200).await.is_err());
    for user in [200, HOST_ID] {
        let token = api.login(user, -100).await;
        assert_eq!(
            api.request(reqwest::Method::POST, "/api/host/query")
                .bearer_auth(&token)
                .json(&serde_json::json!({"view":"overview"}))
                .send()
                .await
                .unwrap()
                .status(),
            403
        );
    }
    let link = api.service.host_link(HOST_ID).await.unwrap();
    let launch = link
        .query_pairs()
        .find(|(key, _)| key == "startapp")
        .unwrap()
        .1
        .into_owned();
    let raw = signed(
        &api.runtime.config.bot_token,
        HOST_ID,
        &launch,
        Utc::now().timestamp(),
    );
    let impostor = signed(
        &api.runtime.config.bot_token,
        200,
        &launch,
        Utc::now().timestamp(),
    );
    assert_eq!(api.login_raw(&impostor).await.status(), 401);
    let response = api.login_raw(&raw).await;
    assert_eq!(response.status(), 200);
    let session: serde_json::Value = response.json().await.unwrap();
    assert_eq!(session["scope"], "host");
    let token = session["token"].as_str().unwrap();
    assert_eq!(api.login_raw(&raw).await.status(), 401);
    assert_eq!(
        api.save(token, 0, serde_json::json!({"netban":true}))
            .await
            .status(),
        403
    );
    for view in ["overview", "cases", "groups", "people", "audit", "queue"] {
        let response = api
            .request(reqwest::Method::POST, "/api/host/query")
            .bearer_auth(token)
            .json(&serde_json::json!({"view":view}))
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body = response.text().await.unwrap();
        assert_eq!(status, 200, "{view}: {body}");
    }
    let logout = api
        .request(reqwest::Method::POST, "/api/miniapp/logout")
        .bearer_auth(token)
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(logout.status(), 200);
    assert_eq!(
        api.request(reqwest::Method::POST, "/api/host/query")
            .bearer_auth(token)
            .json(&serde_json::json!({"view":"overview"}))
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
}

#[tokio::test]
async fn host_group_links_preserve_group_scope_and_recheck_admin_rights() {
    let api = TestApi::new().await;
    let host_link = api.service.host_link(HOST_ID).await.unwrap();
    let launch = host_link
        .query_pairs()
        .find(|(key, _)| key == "startapp")
        .unwrap()
        .1
        .into_owned();
    let response = api
        .login_raw(&signed(
            &api.runtime.config.bot_token,
            HOST_ID,
            &launch,
            Utc::now().timestamp(),
        ))
        .await;
    let session: serde_json::Value = response.json().await.unwrap();
    let host_token = session["token"].as_str().unwrap();
    let request = |token: &str, chat_id| {
        api.request(reqwest::Method::POST, "/api/host/group-link")
            .bearer_auth(token)
            .json(&serde_json::json!({"chat_id":chat_id}))
    };
    for user in [200, HOST_ID] {
        let token = api.login(user, -100).await;
        assert_eq!(request(&token, -100).send().await.unwrap().status(), 403);
    }
    for chat in [0, HOST_ID, i64::MIN] {
        assert_eq!(
            request(host_token, chat).send().await.unwrap().status(),
            400
        );
    }
    let response = request(host_token, -100).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    let url = Url::parse(body["url"].as_str().unwrap()).unwrap();
    let launch = url
        .query_pairs()
        .find(|(key, _)| key == "startapp")
        .unwrap()
        .1
        .into_owned();
    let raw = signed(
        &api.runtime.config.bot_token,
        HOST_ID,
        &launch,
        Utc::now().timestamp(),
    );
    assert_eq!(
        api.login_raw(&signed(
            &api.runtime.config.bot_token,
            200,
            &launch,
            Utc::now().timestamp()
        ))
        .await
        .status(),
        401
    );
    let response = api.login_raw(&raw).await;
    assert_eq!(response.status(), 200);
    let session: serde_json::Value = response.json().await.unwrap();
    assert_eq!(session["scope"], "group");
    let group_token = session["token"].as_str().unwrap();
    assert_eq!(api.login_raw(&raw).await.status(), 401);
    assert_eq!(
        request(group_token, -200).send().await.unwrap().status(),
        403
    );
    assert_eq!(
        api.request(reqwest::Method::POST, "/api/host/query")
            .bearer_auth(group_token)
            .json(&serde_json::json!({"view":"overview"}))
            .send()
            .await
            .unwrap()
            .status(),
        403
    );
    assert_eq!(
        api.save(
            group_token,
            0,
            serde_json::json!({"ot_template":"{user} 請留意主題。"})
        )
        .await
        .status(),
        200
    );
    assert_eq!(
        api.runtime
            .get_warn_settings(-100)
            .await
            .unwrap()
            .ot_template
            .as_deref(),
        Some("{user} 請留意主題。")
    );
    assert!(api
        .runtime
        .get_warn_settings(-200)
        .await
        .unwrap()
        .ot_template
        .is_none());

    // Issue another link, then remove the host's group rights before it is opened.
    let pending: serde_json::Value = request(host_token, -100)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let url = Url::parse(pending["url"].as_str().unwrap()).unwrap();
    let launch = url
        .query_pairs()
        .find(|(key, _)| key == "startapp")
        .unwrap()
        .1
        .into_owned();
    api.telegram.members.lock().unwrap().insert((-100,HOST_ID),serde_json::json!({"status":"member","user":{"id":HOST_ID,"is_bot":false,"first_name":"Member"}}));
    assert_eq!(
        api.login_raw(&signed(
            &api.runtime.config.bot_token,
            HOST_ID,
            &launch,
            Utc::now().timestamp()
        ))
        .await
        .status(),
        403
    );
    assert_eq!(
        api.save(group_token, 1, serde_json::json!({"ot_template":"changed"}))
            .await
            .status(),
        403
    );
    let denied = request(host_token, -100).send().await.unwrap();
    assert_eq!(denied.status(), 403);
    assert_eq!(
        denied.json::<serde_json::Value>().await.unwrap()["error"],
        "group_access_denied"
    );
    api.telegram
        .members
        .lock()
        .unwrap()
        .remove(&(-100, HOST_ID));
    api.telegram.members.lock().unwrap().insert(
        (-100, 999),
        serde_json::json!({"status":"member","user":{"id":999,"is_bot":true,"first_name":"Bot"}}),
    );
    assert_eq!(
        request(host_token, -100).send().await.unwrap().status(),
        403
    );
    api.telegram.members.lock().unwrap().remove(&(-100, 999));
    api.runtime
        .set_group_banned(-100, true, "test", Some(HOST_ID))
        .await
        .unwrap();
    assert_eq!(
        request(host_token, -100).send().await.unwrap().status(),
        403
    );
    // Losing access to one group must not terminate the separate host session.
    assert_eq!(
        api.request(reqwest::Method::POST, "/api/host/query")
            .bearer_auth(host_token)
            .json(&serde_json::json!({"view":"overview"}))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
}

#[tokio::test]
async fn host_queries_paginate_filter_and_redact_private_diagnostics() {
    let runtime = test_runtime().await;
    for user in 100..127 {
        let mut case = dummy_case(ActionKind::AutoBan, -100, user, Utc::now());
        case.evidence_text = format!("{} test", runtime.config.bot_token);
        runtime.persist_case(&case).await.unwrap();
    }
    let query = |view: &str, search: &str, offset| crate::host_panel::Query {
        view: view.into(),
        search: search.into(),
        offset,
        filter: String::new(),
    };
    let first = runtime.host_query(query("cases", "", 0)).await.unwrap();
    assert_eq!(first["items"].as_array().unwrap().len(), 25);
    assert_eq!(first["has_more"], true);
    assert!(!first.to_string().contains(&runtime.config.bot_token));
    let second = runtime.host_query(query("cases", "", 25)).await.unwrap();
    assert_eq!(second["items"].as_array().unwrap().len(), 2);
    assert_eq!(second["has_more"], false);
    let filtered = runtime.host_query(query("cases", "100", 0)).await.unwrap();
    assert_eq!(filtered["items"].as_array().unwrap().len(), 1);
    assert_eq!(filtered["items"][0]["target_user_id"], 100);
    assert!(runtime.host_query(query("config", "", 0)).await.is_err());
    assert!(runtime
        .host_query(query("cases", "", 100_001))
        .await
        .is_err());
    let escaped = runtime
        .host_query(query("cases", "' OR 1=1 --", 0))
        .await
        .unwrap();
    assert!(escaped["items"].as_array().unwrap().is_empty());
    let mut pending = dummy_case(ActionKind::PendingReport, -100, 300, Utc::now());
    pending.status = "pending_review".into();
    runtime.persist_case(&pending).await.unwrap();
    let mut pending_query = query("cases", "", 0);
    pending_query.filter = "pending_review".into();
    let pending_rows = runtime.host_query(pending_query).await.unwrap();
    assert_eq!(pending_rows["items"].as_array().unwrap().len(), 1);
    assert_eq!(pending_rows["items"][0]["id"], pending.id);
    let mut notice = dummy_case(ActionKind::AutoBan, -100, 400, Utc::now());
    notice.evidence_text = "@ExampleBot".into();
    runtime.capture_rules(&notice).await.unwrap();
    runtime
        .with_conn(|conn| {
            conn.execute("UPDATE rule_notice_jobs SET last_error='timeout'", [])?;
            Ok(())
        })
        .await
        .unwrap();
    let mut failures = query("queue", "", 0);
    failures.filter = "failed".into();
    let failed = runtime.host_query(failures).await.unwrap();
    assert_eq!(failed["items"].as_array().unwrap().len(), 1);
    let summary = runtime.host_query(query("overview", "", 0)).await.unwrap();
    assert_eq!(summary["items"][0]["failed_work"], 1);
    assert_eq!(summary["items"][0]["pending_reports"], 1);
    let mut network = query("queue", "", 0);
    network.filter = "network".into();
    assert!(runtime.host_query(network).await.unwrap()["items"]
        .as_array()
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn role_api_rejects_group_tokens_and_checks_revision_and_target() {
    let api = TestApi::new().await;
    let group_token = api.login(HOST_ID, -100).await;
    let body = serde_json::json!({"request_id":Uuid::new_v4().to_string(),"user_id":300,"role":"maintainer","enabled":true,"expected_revision":0});
    for method in [reqwest::Method::POST, reqwest::Method::PATCH] {
        assert_eq!(
            api.request(method, "/api/host/role")
                .bearer_auth(&group_token)
                .json(&body)
                .send()
                .await
                .unwrap()
                .status(),
            403
        );
    }
    let link = api.service.host_link(HOST_ID).await.unwrap();
    let launch = link
        .query_pairs()
        .find(|(key, _)| key == "startapp")
        .unwrap()
        .1
        .into_owned();
    let response = api
        .login_raw(&signed(
            &api.runtime.config.bot_token,
            HOST_ID,
            &launch,
            Utc::now().timestamp(),
        ))
        .await;
    let session: serde_json::Value = response.json().await.unwrap();
    let token = session["token"].as_str().unwrap();
    let read = api
        .request(reqwest::Method::POST, "/api/host/role")
        .bearer_auth(token)
        .json(&serde_json::json!({"user_id":300}))
        .send()
        .await
        .unwrap();
    assert_eq!(read.status(), 200);
    assert_eq!(
        read.json::<serde_json::Value>().await.unwrap()["maintainer"],
        false
    );
    let write = api
        .request(reqwest::Method::PATCH, "/api/host/role")
        .bearer_auth(token)
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(write.status(), 200);
    assert!(api.runtime.is_maintainer(300).await);
    let mut stale = body.clone();
    stale["request_id"] = serde_json::json!(Uuid::new_v4().to_string());
    stale["enabled"] = serde_json::json!(false);
    assert_eq!(
        api.request(reqwest::Method::PATCH, "/api/host/role")
            .bearer_auth(token)
            .json(&stale)
            .send()
            .await
            .unwrap()
            .status(),
        409
    );
    stale["user_id"] = serde_json::json!(HOST_ID);
    assert_eq!(
        api.request(reqwest::Method::PATCH, "/api/host/role")
            .bearer_auth(token)
            .json(&stale)
            .send()
            .await
            .unwrap()
            .status(),
        403
    );
    assert_eq!(
        api.request(reqwest::Method::POST, "/api/host/role")
            .bearer_auth(token)
            .json(&serde_json::json!({"user_id":-100}))
            .send()
            .await
            .unwrap()
            .status(),
        400
    );
    stale["actor_id"] = serde_json::json!(HOST_ID);
    assert_eq!(
        api.request(reqwest::Method::PATCH, "/api/host/role")
            .bearer_auth(token)
            .json(&stale)
            .send()
            .await
            .unwrap()
            .status(),
        400
    );
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
        api.save(&token, 0, serde_json::json!({"threshold_override":0.5}))
            .await
            .status(),
        403
    );
    assert!(api
        .runtime
        .get_group_modules(-100)
        .await
        .unwrap()
        .spam_threshold_override
        .is_none());
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
