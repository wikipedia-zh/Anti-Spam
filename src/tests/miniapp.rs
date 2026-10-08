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
async fn model_rebuild_api_keeps_host_scope_and_replays_the_committed_result() {
    let api = TestApi::new().await;
    let group = api.login(HOST_ID, -100).await;
    for method in [reqwest::Method::POST, reqwest::Method::PATCH] {
        assert_eq!(
            api.request(method, "/api/host/model/rebuild")
                .bearer_auth(&group)
                .body("{}")
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
        .find(|(k, _)| k == "startapp")
        .unwrap()
        .1
        .into_owned();
    let session: serde_json::Value = api
        .login_raw(&signed(
            &api.runtime.config.bot_token,
            HOST_ID,
            &launch,
            Utc::now().timestamp(),
        ))
        .await
        .json()
        .await
        .unwrap();
    let token = session["token"].as_str().unwrap();
    assert_eq!(
        api.request(reqwest::Method::POST, "/api/host/model/rebuild")
            .bearer_auth(token)
            .body("{\"actor_id\":200}")
            .send()
            .await
            .unwrap()
            .status(),
        400
    );
    let mut revision = String::new();
    loop {
        let response = api
            .request(reqwest::Method::POST, "/api/host/model/rebuild")
            .bearer_auth(token)
            .body("{}")
            .send()
            .await
            .unwrap();
        if response.status() == 429 {
            tokio::time::sleep(Duration::from_millis(5)).await;
            continue;
        }
        assert_eq!(response.status(), 200);
        let preview: serde_json::Value = response.json().await.unwrap();
        revision.push_str(preview["revision"].as_str().unwrap());
        break;
    }
    let payload =
        serde_json::json!({"request_id":Uuid::new_v4().to_string(),"expected_revision":revision});
    let mut saved = None;
    for _ in 0..2 {
        loop {
            let response = api
                .request(reqwest::Method::PATCH, "/api/host/model/rebuild")
                .bearer_auth(token)
                .json(&payload)
                .send()
                .await
                .unwrap();
            if response.status() == 429 {
                tokio::time::sleep(Duration::from_millis(5)).await;
                continue;
            }
            assert_eq!(response.status(), 200);
            let result: serde_json::Value = response.json().await.unwrap();
            if let Some(previous) = &saved {
                assert_eq!(previous, &result);
            }
            saved = Some(result);
            break;
        }
    }
    let mut forged = payload;
    forged["actor_id"] = serde_json::json!(HOST_ID);
    assert_eq!(
        api.request(reqwest::Method::PATCH, "/api/host/model/rebuild")
            .bearer_auth(token)
            .json(&forged)
            .send()
            .await
            .unwrap()
            .status(),
        400
    );
    assert!(api
        .telegram
        .requests
        .lock()
        .unwrap()
        .iter()
        .all(|(m, _)| m != "banchatmember" && m != "sendmessage"));
}

#[tokio::test]
async fn model_checks_require_host_scope_and_do_not_call_telegram() {
    let api = TestApi::new().await;
    let group = api.login(HOST_ID, -100).await;
    assert_eq!(
        api.request(reqwest::Method::POST, "/api/host/model")
            .bearer_auth(group)
            .body("{\"action\":\"summary\"}")
            .send()
            .await
            .unwrap()
            .status(),
        403
    );
    let link = api.service.host_link(HOST_ID).await.unwrap();
    let launch = link
        .query_pairs()
        .find(|(k, _)| k == "startapp")
        .unwrap()
        .1
        .into_owned();
    let session: serde_json::Value = api
        .login_raw(&signed(
            &api.runtime.config.bot_token,
            HOST_ID,
            &launch,
            Utc::now().timestamp(),
        ))
        .await
        .json()
        .await
        .unwrap();
    let token = session["token"].as_str().unwrap();
    for body in [
        serde_json::json!({"action":"summary","actor_id":200}),
        serde_json::json!({"action":"train","text":"hi"}),
        serde_json::json!({"action":"score","text":"字".repeat(4001)}),
    ] {
        assert_eq!(
            api.request(reqwest::Method::POST, "/api/host/model")
                .bearer_auth(token)
                .json(&body)
                .send()
                .await
                .unwrap()
                .status(),
            400
        );
    }
    for body in [
        serde_json::json!({"action":"summary"}),
        serde_json::json!({"action":"score","text":"hello"}),
        serde_json::json!({"action":"evaluate"}),
    ] {
        loop {
            let response = api
                .request(reqwest::Method::POST, "/api/host/model")
                .bearer_auth(token)
                .json(&body)
                .send()
                .await
                .unwrap();
            if response.status() == 429 {
                tokio::time::sleep(Duration::from_millis(5)).await;
                continue;
            }
            assert_eq!(response.status(), 200);
            assert_eq!(response.headers()["cache-control"], "no-store");
            break;
        }
    }
    assert!(api
        .telegram
        .requests
        .lock()
        .unwrap()
        .iter()
        .all(|(m, _)| m != "banchatmember" && m != "sendmessage"));
}

#[tokio::test]
async fn rule_management_requires_host_scope_and_trials_never_touch_telegram() {
    let api = TestApi::new().await;
    let group = api.login(HOST_ID, -100).await;
    for (path, method) in [
        ("/api/host/rule", reqwest::Method::POST),
        ("/api/host/rule", reqwest::Method::PATCH),
        ("/api/host/rule/test", reqwest::Method::POST),
    ] {
        assert_eq!(
            api.request(method, path)
                .bearer_auth(&group)
                .body("{}")
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
        .find(|(k, _)| k == "startapp")
        .unwrap()
        .1
        .into_owned();
    let session: serde_json::Value = api
        .login_raw(&signed(
            &api.runtime.config.bot_token,
            HOST_ID,
            &launch,
            Utc::now().timestamp(),
        ))
        .await
        .json()
        .await
        .unwrap();
    let token = session["token"].as_str().unwrap();
    let body = serde_json::json!({"request_id":Uuid::new_v4().to_string(),"rule_id":null,"expected_revision":0,"rule":{"pattern":"casino","description":"test"}});
    let mut forged = body.clone();
    forged["actor_id"] = serde_json::json!(200);
    assert_eq!(
        api.request(reqwest::Method::PATCH, "/api/host/rule")
            .bearer_auth(token)
            .body(forged.to_string())
            .send()
            .await
            .unwrap()
            .status(),
        400
    );
    for (path, method, payload) in [
        (
            "/api/host/rule/test",
            reqwest::Method::POST,
            serde_json::json!({"pattern":"casino","text":"casino"}),
        ),
        ("/api/host/rule", reqwest::Method::PATCH, body.clone()),
        ("/api/host/rule", reqwest::Method::PATCH, body),
    ] {
        loop {
            let result = api
                .request(method.clone(), path)
                .bearer_auth(token)
                .body(payload.to_string())
                .send()
                .await
                .unwrap();
            if result.status() == 429 {
                tokio::task::yield_now().await;
                continue;
            }
            assert_eq!(result.status(), 200);
            break;
        }
    }
    assert_eq!(api.runtime.list_spam_rules().await.unwrap().len(), 1);
    assert!(api
        .telegram
        .requests
        .lock()
        .unwrap()
        .iter()
        .all(|(method, _)| method != "banchatmember" && method != "sendmessage"));
}

#[tokio::test]
async fn host_review_api_checks_scope_target_and_replays_the_saved_decision() {
    let api = TestApi::new().await;
    let mut case = dummy_case(ActionKind::PendingReport, -100, 300, Utc::now());
    case.status = "pending_review".into();
    api.runtime.persist_case(&case).await.unwrap();
    let group = api.login(HOST_ID, -100).await;
    let send = |token: String, body: serde_json::Value| {
        api.request(reqwest::Method::PATCH, "/api/host/case/review")
            .bearer_auth(token)
            .json(&body)
            .send()
    };
    assert_eq!(
        send(group, serde_json::json!({})).await.unwrap().status(),
        403
    );
    let link = api.service.host_link(HOST_ID).await.unwrap();
    let launch = link
        .query_pairs()
        .find(|(key, _)| key == "startapp")
        .unwrap()
        .1
        .into_owned();
    let session: serde_json::Value = api
        .login_raw(&signed(
            &api.runtime.config.bot_token,
            HOST_ID,
            &launch,
            Utc::now().timestamp(),
        ))
        .await
        .json()
        .await
        .unwrap();
    let token = session["token"].as_str().unwrap().to_string();
    let snapshot = api
        .runtime
        .host_case(crate::host_cases::Query {
            case_id: case.id.clone(),
            offset: 0,
        })
        .await
        .unwrap()
        .unwrap();
    let mut body = serde_json::json!({"request_id":Uuid::new_v4().to_string(),"case_id":case.id,"target_user_id":999,"expected_revision":snapshot["revision"],"kind":"report","decision":"approve"});
    assert_eq!(
        send(token.clone(), body.clone()).await.unwrap().status(),
        409
    );
    body["target_user_id"] = serde_json::json!(300);
    let mut forged = body.clone();
    forged["actor_id"] = serde_json::json!(200);
    assert_eq!(send(token.clone(), forged).await.unwrap().status(), 400);
    let saved = send(token.clone(), body.clone()).await.unwrap();
    assert_eq!(saved.status(), 200);
    assert_eq!(
        saved.json::<serde_json::Value>().await.unwrap()["status"],
        "ban_pending"
    );
    assert_eq!(
        send(token.clone(), body.clone()).await.unwrap().status(),
        200
    );
    body["request_id"] = serde_json::json!(Uuid::new_v4().to_string());
    assert_eq!(send(token, body).await.unwrap().status(), 409);
    assert!(!api
        .telegram
        .requests
        .lock()
        .unwrap()
        .iter()
        .any(|(m, _)| m == "banchatmember"));
}

#[tokio::test]
async fn host_case_api_enforces_scope_target_and_confirmation_revision() {
    let api = TestApi::new().await;
    let case = dummy_case(ActionKind::AutoBan, -100, 300, Utc::now());
    api.runtime.persist_case(&case).await.unwrap();
    let group = api.login(HOST_ID, -100).await;
    for (method, path) in [
        (reqwest::Method::POST, "/api/host/case"),
        (reqwest::Method::PATCH, "/api/host/case/reverse"),
    ] {
        assert_eq!(
            api.request(method, path)
                .bearer_auth(&group)
                .json(&serde_json::json!({"case_id":case.id}))
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
    let session: serde_json::Value = api
        .login_raw(&signed(
            &api.runtime.config.bot_token,
            HOST_ID,
            &launch,
            Utc::now().timestamp(),
        ))
        .await
        .json()
        .await
        .unwrap();
    let token = session["token"].as_str().unwrap();
    let read = api
        .request(reqwest::Method::POST, "/api/host/case")
        .bearer_auth(token)
        .json(&serde_json::json!({"case_id":case.id}))
        .send()
        .await
        .unwrap();
    assert_eq!(read.status(), 200);
    let snapshot: serde_json::Value = read.json().await.unwrap();
    let mut body = serde_json::json!({"case_id":case.id,"target_user_id":999,"request_id":Uuid::new_v4().to_string(),"expected_revision":snapshot["revision"]});
    let write = |body: &serde_json::Value| {
        api.request(reqwest::Method::PATCH, "/api/host/case/reverse")
            .bearer_auth(token)
            .json(body)
    };
    assert_eq!(write(&body).send().await.unwrap().status(), 409);
    body["target_user_id"] = serde_json::json!(300);
    let mut injected = body.clone();
    injected["actor_id"] = serde_json::json!(123);
    assert_eq!(write(&injected).send().await.unwrap().status(), 400);
    assert_eq!(write(&body).send().await.unwrap().status(), 200);
    assert_eq!(write(&body).send().await.unwrap().status(), 200);
    body["request_id"] = serde_json::json!(Uuid::new_v4().to_string());
    assert_eq!(write(&body).send().await.unwrap().status(), 409);
    assert_eq!(
        api.request(reqwest::Method::POST, "/api/host/case")
            .bearer_auth(token)
            .json(&serde_json::json!({"case_id":"missing"}))
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    assert!(!api
        .telegram
        .requests
        .lock()
        .unwrap()
        .iter()
        .any(|(method, _)| method == "unbanchatmember"));
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
        created_from: None,
        created_before: None,
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
async fn host_date_filters_include_full_days_and_keep_pagination_inside_the_range() {
    let runtime = test_runtime().await;
    let timestamp = |s: &str| DateTime::parse_from_rfc3339(s).unwrap().timestamp();
    let from = timestamp("2026-10-08T00:00:00+08:00");
    let before = timestamp("2026-10-09T00:00:00+08:00");
    let dates = [
        "2026-10-07T23:59:59.999999+08:00",
        "2026-10-08T00:00:00+08:00",
        "2026-10-08T23:59:59.999999+08:00",
        "2026-10-09T00:00:00+08:00",
    ];
    for (n, date) in dates.iter().enumerate() {
        let mut case = dummy_case(ActionKind::AutoBan, -100, 200 + n as i64, Utc::now());
        case.created_at = DateTime::parse_from_rfc3339(date)
            .unwrap()
            .with_timezone(&Utc);
        runtime.persist_case(&case).await.unwrap();
    }
    // Mix offset timestamps and UTC timestamps, including sub-millisecond values.
    for user in 300..325 {
        let mut case = dummy_case(ActionKind::AutoBan, -100, user, Utc::now());
        case.created_at = DateTime::parse_from_rfc3339("2026-10-08T10:00:00.999999Z")
            .unwrap()
            .with_timezone(&Utc);
        runtime.persist_case(&case).await.unwrap();
    }
    let query = |view: &str, offset| {
        serde_json::from_value::<crate::host_panel::Query>(serde_json::json!({"view":view,"offset":offset,"created_from":from,"created_before":before})).unwrap()
    };
    let first = runtime.host_query(query("cases", 0)).await.unwrap();
    assert_eq!(first["items"].as_array().unwrap().len(), 25);
    assert_eq!(first["has_more"], true);
    let last = runtime.host_query(query("cases", 25)).await.unwrap();
    assert_eq!(last["has_more"], false);
    let ids: Vec<_> = last["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["target_user_id"].as_i64().unwrap())
        .collect();
    assert_eq!(ids, [202, 201]);
    let mut only_user = query("cases", 0);
    only_user.search = "202".into();
    assert_eq!(
        runtime.host_query(only_user).await.unwrap()["items"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let mut before_only = query("cases", 0);
    before_only.created_from = None;
    assert_eq!(
        runtime.host_query(before_only).await.unwrap()["has_more"],
        true
    );
    runtime.with_conn(move |conn| {
        for date in dates {
            conn.execute("INSERT INTO maintainer_actions(actor_id,actor_name,chat_id,command,summary,undo_data,created_at) VALUES (?1,'host',-100,'test','test',?2,?3)", params![HOST_ID,serde_json::to_string(&UndoData::NotRevertible)?,date])?;
        }
        Ok(())
    }).await.unwrap();
    let audit = runtime.host_query(query("audit", 0)).await.unwrap();
    assert_eq!(audit["items"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn host_result_filters_do_not_treat_pending_or_uncertain_actions_as_done() {
    let runtime = test_runtime().await;
    let cases = [
        (ActionKind::AutoBan, "ban_pending"),
        (ActionKind::AutoBan, "ban_failed"),
        (ActionKind::AutoBan, "banned_delete_failed"),
        (ActionKind::AutoBan, "auto_banned"),
        (ActionKind::SpamBan, "done"),
        (ActionKind::Mute, "done"),
        (ActionKind::Mute, "action_unconfirmed"),
        (ActionKind::Mute, "action_failed"),
        (ActionKind::Mute, "action_pending"),
        (ActionKind::Mute, "action_cancelled"),
        (ActionKind::AutoBan, "reversal_pending"),
        (ActionKind::AutoBan, "reversed"),
        (ActionKind::ReportRejected, "rejected_and_cleaned"),
        (ActionKind::PendingReport, "pending_review"),
    ];
    for (index, (action, status)) in cases.into_iter().enumerate() {
        let mut case = dummy_case(action, -100, 200 + index as i64, Utc::now());
        case.status = status.into();
        runtime.persist_case(&case).await.unwrap();
    }
    for (filter, expected) in [
        ("banned", vec![204, 203, 202]),
        ("pending", vec![208, 200]),
        ("failed", vec![207, 202, 201]),
        ("unconfirmed", vec![206]),
        ("cancelled", vec![209]),
        ("reversal_pending", vec![210]),
        ("reversed", vec![211]),
        ("rejected", vec![212]),
        ("pending_review", vec![213]),
    ] {
        let query =
            serde_json::from_value(serde_json::json!({"view":"cases","filter":filter})).unwrap();
        let rows = runtime.host_query(query).await.unwrap();
        let ids: Vec<_> = rows["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["target_user_id"].as_i64().unwrap())
            .collect();
        assert_eq!(ids, expected, "{filter}");
    }
    for payload in [
        serde_json::json!({"view":"cases","filter":"' OR 1=1 --"}),
        serde_json::json!({"view":"cases","created_from":100,"created_before":100}),
        serde_json::json!({"view":"cases","created_from":101,"created_before":100}),
        serde_json::json!({"view":"cases","created_from":-1}),
        serde_json::json!({"view":"cases","created_before":253402300800_i64}),
        serde_json::json!({"view":"groups","created_from":0}),
    ] {
        assert!(runtime
            .host_query(serde_json::from_value(payload).unwrap())
            .await
            .is_err());
    }
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

#[tokio::test]
async fn operations_api_requires_host_scope_and_rejects_forged_actors() {
    let api=TestApi::new().await;let group=api.login(HOST_ID,-100).await;
    for method in [reqwest::Method::POST,reqwest::Method::PATCH] {assert_eq!(api.request(method,"/api/host/operations").bearer_auth(&group).json(&serde_json::json!({})).send().await.unwrap().status(),403);}
    let link=api.service.host_link(HOST_ID).await.unwrap();let launch=link.query_pairs().find(|(k,_)|k=="startapp").unwrap().1.into_owned();
    let session:serde_json::Value=api.login_raw(&signed(&api.runtime.config.bot_token,HOST_ID,&launch,Utc::now().timestamp())).await.json().await.unwrap();let token=session["token"].as_str().unwrap();
    let body=serde_json::json!({"request_id":Uuid::new_v4().to_string(),"expected_revision":0,"controls":{"automatic_new_paused":true,"automatic_pending_paused":true,"network_paused":true}});
    assert_eq!(api.request(reqwest::Method::POST,"/api/host/operations").bearer_auth(token).json(&serde_json::json!({})).send().await.unwrap().status(),200);
    for method in [reqwest::Method::POST,reqwest::Method::PATCH] {let mut forged=body.clone();forged["actor_id"]=serde_json::json!(HOST_ID);assert_eq!(api.request(method,"/api/host/operations").bearer_auth(token).json(&forged).send().await.unwrap().status(),400);}
    for _ in 0..2 {assert_eq!(api.request(reqwest::Method::PATCH,"/api/host/operations").bearer_auth(token).json(&body).send().await.unwrap().status(),200);}
    assert!(api.runtime.operations_controls().await.unwrap().network_paused);
}

#[tokio::test]
async fn departure_api_requires_host_scope_and_saves_once_without_calling_telegram() {
    let api=TestApi::new().await;let group=api.login(HOST_ID,-100).await;
    api.runtime.record_group_seen(-100,Some("Test")).await;
    for method in [reqwest::Method::POST,reqwest::Method::PATCH] {assert_eq!(api.request(method,"/api/host/group/leave").bearer_auth(&group).json(&serde_json::json!({})).send().await.unwrap().status(),403);}
    let link=api.service.host_link(HOST_ID).await.unwrap();let launch=link.query_pairs().find(|(k,_)|k=="startapp").unwrap().1.into_owned();
    let session:serde_json::Value=api.login_raw(&signed(&api.runtime.config.bot_token,HOST_ID,&launch,Utc::now().timestamp())).await.json().await.unwrap();let token=session["token"].as_str().unwrap();
    let snapshot:serde_json::Value=api.request(reqwest::Method::POST,"/api/host/group/leave").bearer_auth(token).json(&serde_json::json!({"chat_id":-100})).send().await.unwrap().json().await.unwrap();
    let body=serde_json::json!({"request_id":Uuid::new_v4().to_string(),"chat_id":-100,"expected_revision":snapshot["revision"],"reason":"Missing permissions","block_rejoin":false});
    let mut forged=body.clone();forged["actor_id"]=serde_json::json!(HOST_ID);
    assert_eq!(api.request(reqwest::Method::PATCH,"/api/host/group/leave").bearer_auth(token).json(&forged).send().await.unwrap().status(),400);
    for _ in 0..2 {let result=api.request(reqwest::Method::PATCH,"/api/host/group/leave").bearer_auth(token).json(&body).send().await.unwrap();assert_eq!(result.status(),200);let result:serde_json::Value=result.json().await.unwrap();assert_eq!(result["state"],"queued");}
    assert!(!api.runtime.is_group_banned(-100).await);
    assert_eq!(api.telegram.requests.lock().unwrap().iter().filter(|(m,_)|m=="leavechat").count(),0);
}

#[tokio::test]
async fn retry_api_is_host_only_and_never_calls_telegram_directly() {
    let api=TestApi::new().await;let group=api.login(HOST_ID,-100).await;
    let route="/api/host/queue/item";
    for method in [reqwest::Method::POST,reqwest::Method::PATCH] {
        assert_eq!(api.request(method.clone(),route).json(&serde_json::json!({})).send().await.unwrap().status(),401);
        assert_eq!(api.request(method,route).bearer_auth(&group).json(&serde_json::json!({})).send().await.unwrap().status(),403);
    }
    assert_eq!(api.request(reqwest::Method::POST,"/api/host/query").bearer_auth(group).json(&serde_json::json!({"view":"maintenance"})).send().await.unwrap().status(),403);
    let link=api.service.host_link(HOST_ID).await.unwrap();let launch=link.query_pairs().find(|(k,_)|k=="startapp").unwrap().1.into_owned();
    let session:serde_json::Value=api.login_raw(&signed(&api.runtime.config.bot_token,HOST_ID,&launch,Utc::now().timestamp())).await.json().await.unwrap();let token=session["token"].as_str().unwrap();
    let case=dummy_case(ActionKind::AutoBan,-100,200,Utc::now());api.runtime.persist_case(&case).await.unwrap();api.runtime.set_group_module(-300,"netban",true).await.unwrap();api.runtime.enqueue_network_deliveries(&case.id).await.unwrap();
    api.runtime.with_conn(|c|{c.execute("UPDATE network_deliveries SET last_error='failure',next_attempt_at=?1",[Utc::now().timestamp()+3600])?;Ok(())}).await.unwrap();
    let target=serde_json::json!({"kind":"network","case_id":case.id,"chat_id":-300});
    let snapshot:serde_json::Value=api.request(reqwest::Method::POST,route).bearer_auth(token).json(&serde_json::json!({"target":target})).send().await.unwrap().json().await.unwrap();
    let body=serde_json::json!({"request_id":Uuid::new_v4().to_string(),"target":target,"expected_revision":snapshot["revision"]});let before=api.telegram.requests.lock().unwrap().len();
    let mut forged=body.clone();forged["actor_id"]=serde_json::json!(HOST_ID);
    assert_eq!(api.request(reqwest::Method::PATCH,route).bearer_auth(token).json(&forged).send().await.unwrap().status(),400);
    for _ in 0..2 {assert_eq!(api.request(reqwest::Method::PATCH,route).bearer_auth(token).json(&body).send().await.unwrap().status(),200);}
    assert_eq!(api.telegram.requests.lock().unwrap().len(),before);
    let health=api.request(reqwest::Method::POST,"/api/host/query").bearer_auth(token).json(&serde_json::json!({"view":"maintenance"})).send().await.unwrap();assert_eq!(health.status(),200);
    assert_eq!(health.json::<serde_json::Value>().await.unwrap()["items"][0]["backup"]["status"],"missing");
}
