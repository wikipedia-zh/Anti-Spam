use super::*;
use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

type HmacSha256 = Hmac<Sha256>;
const AUTH_TTL: i64 = 300;
const SESSION_TTL: i64 = 900;
const CAPACITY: usize = 4096;

pub(super) struct Settings {
    pub origin: String,
    pub proxy_key: String,
    pub launch_url: Url,
    pub listen: std::net::SocketAddr,
}

impl Settings {
    fn from_env() -> Result<Option<Self>> {
        let Ok(launch) = env::var("MINIAPP_LAUNCH_URL") else {
            return Ok(None);
        };
        let launch_url = Url::parse(&launch).context("invalid MINIAPP_LAUNCH_URL")?;
        anyhow::ensure!(
            launch_url.scheme() == "https"
                && launch_url.host_str() == Some("t.me")
                && launch_url.query().is_none()
                && launch_url.fragment().is_none()
                && launch_url.username().is_empty()
                && launch_url.password().is_none(),
            "MINIAPP_LAUNCH_URL must be an HTTPS Telegram app link without parameters"
        );
        let origin = env::var("MINIAPP_ORIGIN").context("MINIAPP_ORIGIN is required")?;
        let parsed = Url::parse(&origin).context("invalid MINIAPP_ORIGIN")?;
        anyhow::ensure!(
            parsed.scheme() == "https" && parsed.origin().ascii_serialization() == origin,
            "MINIAPP_ORIGIN must be an HTTPS origin without a path"
        );
        let proxy_key = env::var("MINIAPP_PROXY_KEY").context("MINIAPP_PROXY_KEY is required")?;
        anyhow::ensure!(
            proxy_key.len() >= 32 && proxy_key.bytes().all(|b| b.is_ascii_graphic()),
            "MINIAPP_PROXY_KEY must have at least 32 printable characters"
        );
        let listen = env::var("MINIAPP_LISTEN")
            .unwrap_or_else(|_| "127.0.0.1:8081".into())
            .parse()
            .context("invalid MINIAPP_LISTEN")?;
        Ok(Some(Self {
            origin,
            proxy_key,
            launch_url,
            listen,
        }))
    }
}

#[derive(Clone)]
struct Grant {
    user_id: i64,
    chat_id: i64,
    expires_at: i64,
}

struct Rate {
    start: Instant,
    count: u32,
}
impl Default for Rate {
    fn default() -> Self {
        Self {
            start: Instant::now(),
            count: 0,
        }
    }
}
impl Rate {
    fn allow(&mut self, limit: u32) -> bool {
        if self.start.elapsed() >= Duration::from_secs(60) {
            *self = Self::default();
        }
        self.count += 1;
        self.count <= limit
    }
}

#[derive(Default)]
struct Access {
    launches: HashMap<String, Grant>,
    sessions: HashMap<String, Grant>,
    user_rates: HashMap<i64, Rate>,
    global_rate: Rate,
}

pub(super) struct Service {
    settings: Settings,
    access: Mutex<Access>,
    requests: tokio::sync::Semaphore,
}

impl Service {
    pub(super) fn new(settings: Settings) -> Self {
        Self {
            settings,
            access: Mutex::new(Access::default()),
            requests: tokio::sync::Semaphore::new(32),
        }
    }

    pub(super) async fn launch_link(&self, user_id: i64, chat_id: i64) -> Result<Url> {
        let now = Utc::now().timestamp();
        let mut access = self.access.lock().await;
        access.launches.retain(|_, g| g.expires_at > now);
        access.sessions.retain(|_, g| g.expires_at > now);
        access
            .user_rates
            .retain(|_, r| r.start.elapsed() < Duration::from_secs(60));
        anyhow::ensure!(access.launches.len() < CAPACITY, "too many active launches");
        anyhow::ensure!(
            access.user_rates.entry(user_id).or_default().allow(30),
            "too many settings requests"
        );
        let token = Uuid::new_v4().simple().to_string();
        access.launches.insert(
            token.clone(),
            Grant {
                user_id,
                chat_id,
                expires_at: now + AUTH_TTL,
            },
        );
        let mut link = self.settings.launch_url.clone();
        link.query_pairs_mut().append_pair("startapp", &token);
        Ok(link)
    }
}

pub(super) struct InitData {
    pub user_id: i64,
    pub launch: String,
}

pub(super) fn validate_init_data(raw: &str, bot_token: &str, now: i64) -> Result<InitData> {
    anyhow::ensure!(!raw.is_empty() && raw.len() <= 16_384, "invalid init data");
    let mut fields = BTreeMap::new();
    for (key, value) in url::form_urlencoded::parse(raw.as_bytes()) {
        anyhow::ensure!(
            !key.contains(['\n', '\r']) && !value.contains(['\n', '\r']),
            "invalid init field"
        );
        anyhow::ensure!(
            fields
                .insert(key.into_owned(), value.into_owned())
                .is_none(),
            "duplicate init field"
        );
    }
    let hash = fields.remove("hash").context("missing hash")?;
    anyhow::ensure!(
        hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid hash"
    );
    let bytes: Vec<_> = (0..64)
        .step_by(2)
        .map(|i| u8::from_str_radix(&hash[i..i + 2], 16))
        .collect::<std::result::Result<_, _>>()?;
    let check = fields
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("\n");
    let mut secret = HmacSha256::new_from_slice(b"WebAppData")?;
    secret.update(bot_token.as_bytes());
    let mut mac = HmacSha256::new_from_slice(&secret.finalize().into_bytes())?;
    mac.update(check.as_bytes());
    mac.verify_slice(&bytes)
        .map_err(|_| anyhow::anyhow!("invalid signature"))?;
    let date: i64 = fields
        .get("auth_date")
        .context("missing auth date")?
        .parse()?;
    anyhow::ensure!(
        date >= now - AUTH_TTL && date <= now + 30,
        "expired init data"
    );
    #[derive(Deserialize)]
    struct User {
        id: i64,
        #[serde(default)]
        is_bot: bool,
    }
    let user: User = serde_json::from_str(fields.get("user").context("missing user")?)?;
    anyhow::ensure!(user.id > 0 && !user.is_bot, "invalid user");
    let launch = fields
        .get("start_param")
        .context("missing launch reference")?
        .clone();
    anyhow::ensure!(
        launch.len() == 32 && launch.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid launch reference"
    );
    Ok(InitData {
        user_id: user.id,
        launch,
    })
}

#[derive(Clone)]
pub(super) struct Api {
    pub bot: Bot,
    pub runtime: Arc<Runtime>,
    pub service: Arc<Service>,
}

struct ApiError(StatusCode, &'static str);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({"error":self.1}))).into_response()
    }
}
fn unauthorized() -> ApiError {
    ApiError(StatusCode::UNAUTHORIZED, "session_expired")
}
fn forbidden() -> ApiError {
    ApiError(StatusCode::FORBIDDEN, "forbidden")
}
fn unavailable() -> ApiError {
    ApiError(StatusCode::SERVICE_UNAVAILABLE, "temporarily_unavailable")
}
fn storage_error(error: anyhow::Error) -> ApiError {
    log::error!("miniapp settings storage failed: {error}");
    ApiError(StatusCode::INTERNAL_SERVER_ERROR, "save_failed")
}

async fn boundary(State(api): State<Api>, request: Request, next: Next) -> Response {
    let request_id = Uuid::new_v4().to_string();
    let origin = request
        .headers()
        .get("origin")
        .and_then(|v| v.to_str().ok());
    let key = request
        .headers()
        .get("x-spb-proxy-key")
        .map(|v| v.as_bytes())
        .unwrap_or(&[]);
    let mut signature =
        HmacSha256::new_from_slice(api.service.settings.proxy_key.as_bytes()).expect("HMAC key");
    signature.update(b"spb-miniapp-proxy");
    let mut supplied = HmacSha256::new_from_slice(key).expect("HMAC key");
    supplied.update(b"spb-miniapp-proxy");
    let trusted = request.headers().get_all("origin").iter().count() == 1
        && request.headers().get_all("x-spb-proxy-key").iter().count() == 1
        && supplied
            .verify_slice(&signature.finalize().into_bytes())
            .is_ok();
    let mut response = if !trusted || origin != Some(api.service.settings.origin.as_str()) {
        forbidden().into_response()
    } else if !api.service.access.lock().await.global_rate.allow(600) {
        ApiError(StatusCode::TOO_MANY_REQUESTS, "rate_limited").into_response()
    } else if let Ok(_permit) = api.service.requests.try_acquire() {
        match tokio::time::timeout(Duration::from_secs(20), next.run(request)).await {
            Ok(response) => response,
            Err(_) => unavailable().into_response(),
        }
    } else {
        unavailable().into_response()
    };
    response
        .headers_mut()
        .insert("cache-control", "no-store".parse().unwrap());
    response
        .headers_mut()
        .insert("x-content-type-options", "nosniff".parse().unwrap());
    response
        .headers_mut()
        .insert("x-request-id", request_id.parse().unwrap());
    response
}

async fn permissions(api: &Api, grant: &Grant) -> std::result::Result<bool, ApiError> {
    if api.runtime.is_group_banned(grant.chat_id).await
        || (api.runtime.is_user_banned(grant.user_id).await
            && !api.runtime.is_maintainer(grant.user_id).await)
    {
        return Err(forbidden());
    }
    let check = async {
        let chat = api
            .bot
            .get_chat(ChatId(grant.chat_id))
            .await
            .map_err(|_| unavailable())?;
        if !chat.is_group() && !chat.is_supergroup() {
            return Err(forbidden());
        }
        let user = api
            .bot
            .get_chat_member(ChatId(grant.chat_id), UserId(grant.user_id as u64))
            .await
            .map_err(|_| unavailable())?;
        if !user.kind.is_privileged() {
            return Err(forbidden());
        }
        let bot_id = api.runtime.me_id(&api.bot).await.ok_or_else(unavailable)?;
        let member = api
            .bot
            .get_chat_member(ChatId(grant.chat_id), bot_id)
            .await
            .map_err(|_| unavailable())?;
        if !member.kind.is_privileged() {
            return Err(forbidden());
        }
        Ok(())
    };
    tokio::time::timeout(Duration::from_secs(10), check)
        .await
        .map_err(|_| unavailable())??;
    if grant.expires_at <= Utc::now().timestamp() {
        return Err(unauthorized());
    }
    Ok(api.runtime.is_maintainer(grant.user_id).await
        && api.runtime.config.test_group_id != Some(grant.chat_id)
        && api.runtime.project_chat().await != Some(grant.chat_id))
}

fn token_hash(token: &str) -> String {
    format!("{:x}", Sha256::digest(token.as_bytes()))
}

async fn session(api: &Api, headers: &HeaderMap) -> std::result::Result<Grant, ApiError> {
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .filter(|s| s.len() == 32)
        .ok_or_else(unauthorized)?;
    let mut access = api.service.access.lock().await;
    let now = Utc::now().timestamp();
    access.sessions.retain(|_, g| g.expires_at > now);
    let grant = access
        .sessions
        .get(&token_hash(token))
        .cloned()
        .ok_or_else(unauthorized)?;
    if !access
        .user_rates
        .entry(grant.user_id)
        .or_default()
        .allow(60)
    {
        return Err(ApiError(StatusCode::TOO_MANY_REQUESTS, "rate_limited"));
    }
    Ok(grant)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Login {
    init_data: String,
}

async fn login(
    State(api): State<Api>,
    payload: std::result::Result<Json<Login>, axum::extract::rejection::JsonRejection>,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    let Json(payload) =
        payload.map_err(|_| ApiError(StatusCode::BAD_REQUEST, "invalid_request"))?;
    let now = Utc::now().timestamp();
    let init = validate_init_data(&payload.init_data, &api.runtime.config.bot_token, now)
        .map_err(|_| unauthorized())?;
    let grant = {
        let mut access = api.service.access.lock().await;
        access.launches.retain(|_, g| g.expires_at > now);
        let grant = access
            .launches
            .get(&init.launch)
            .filter(|g| g.user_id == init.user_id)
            .cloned()
            .ok_or_else(unauthorized)?;
        if !access
            .user_rates
            .entry(grant.user_id)
            .or_default()
            .allow(30)
        {
            return Err(ApiError(StatusCode::TOO_MANY_REQUESTS, "rate_limited"));
        }
        grant
    };
    permissions(&api, &grant).await?;
    let mut access = api.service.access.lock().await;
    access.sessions.retain(|_, g| g.expires_at > now);
    if access.sessions.len() >= CAPACITY {
        return Err(unavailable());
    }
    // Consume after authorization; competing requests cannot redeem it twice.
    if access.launches.remove(&init.launch).is_none() {
        return Err(unauthorized());
    }
    let token = Uuid::new_v4().simple().to_string();
    let expires_at = Utc::now().timestamp() + SESSION_TTL;
    access.sessions.insert(
        token_hash(&token),
        Grant {
            expires_at,
            ..grant
        },
    );
    Ok(Json(
        serde_json::json!({"token":token,"expires_at":expires_at}),
    ))
}

async fn read_settings(
    State(api): State<Api>,
    headers: HeaderMap,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    let grant = session(&api, &headers).await?;
    let can_edit_threshold = permissions(&api, &grant).await?;
    let settings = api
        .runtime
        .group_settings_snapshot(grant.chat_id)
        .await
        .map_err(storage_error)?;
    let threshold = api
        .runtime
        .current_threshold()
        .await
        .map_err(storage_error)?;
    Ok(Json(
        serde_json::json!({"settings":settings,"global_threshold":threshold,"can_edit_threshold":can_edit_threshold,"expires_at":grant.expires_at}),
    ))
}

async fn save_settings(
    State(api): State<Api>,
    headers: HeaderMap,
    payload: std::result::Result<
        Json<group_settings::Patch>,
        axum::extract::rejection::JsonRejection,
    >,
) -> std::result::Result<Json<group_settings::Snapshot>, ApiError> {
    let grant = session(&api, &headers).await?;
    let can_edit_threshold = permissions(&api, &grant).await?;
    let Json(patch) = payload.map_err(|_| ApiError(StatusCode::BAD_REQUEST, "invalid_request"))?;
    match api
        .runtime
        .save_group_settings(grant.chat_id, grant.user_id, patch, can_edit_threshold)
        .await
        .map_err(storage_error)?
    {
        group_settings::SaveResult::Saved(settings) => Ok(Json(settings)),
        group_settings::SaveResult::Conflict => {
            Err(ApiError(StatusCode::CONFLICT, "settings_changed"))
        }
        group_settings::SaveResult::Invalid => {
            Err(ApiError(StatusCode::BAD_REQUEST, "invalid_settings"))
        }
        group_settings::SaveResult::Forbidden => Err(forbidden()),
    }
}

pub(super) fn router(api: Api) -> Router {
    Router::new()
        .route("/api/miniapp/session", post(login))
        .route(
            "/api/groups/current/settings",
            get(read_settings).patch(save_settings),
        )
        .layer(DefaultBodyLimit::max(20_480))
        .layer(middleware::from_fn_with_state(api.clone(), boundary))
        .with_state(api)
}

pub(super) struct Server {
    shutdown: tokio::sync::oneshot::Sender<()>,
    task: tokio::task::JoinHandle<()>,
}
impl Server {
    pub(super) async fn stop(mut self) {
        let _ = self.shutdown.send(());
        if tokio::time::timeout(Duration::from_secs(25), &mut self.task)
            .await
            .is_err()
        {
            self.task.abort();
        }
    }
}

pub(super) async fn start(bot: Bot, runtime: Arc<Runtime>) -> Result<Option<Server>> {
    let Some(settings) = Settings::from_env()? else {
        return Ok(None);
    };
    let me = tokio::time::timeout(Duration::from_secs(10), bot.get_me()).await??;
    let _ = runtime.me_id.set(me.id);
    let username = settings
        .launch_url
        .path_segments()
        .and_then(|mut s| s.next())
        .unwrap_or("");
    anyhow::ensure!(
        me.username
            .as_deref()
            .is_some_and(|name| name.eq_ignore_ascii_case(username)),
        "Mini App link must belong to this bot"
    );
    let listener = tokio::net::TcpListener::bind(settings.listen).await?;
    let service = Arc::new(Service::new(settings));
    runtime
        .miniapp
        .set(service.clone())
        .map_err(|_| anyhow::anyhow!("Mini App already started"))?;
    let app = router(Api {
        bot,
        runtime,
        service,
    });
    let (shutdown, receiver) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = receiver.await;
            })
            .await
        {
            log::error!("miniapp server stopped: {error}");
        }
    });
    Ok(Some(Server { shutdown, task }))
}

pub(super) async fn launch(bot: &Bot, runtime: &Runtime, message: &Message) -> ResponseResult<()> {
    if !message.chat.is_group() && !message.chat.is_supergroup() {
        return reply_ephemeral(bot, message, "請在要設定的群組輸入 /settings。").await;
    }
    let Some(user) = message.from.as_ref() else {
        return Ok(());
    };
    if !is_group_admin(bot, message.chat.id, user.id.0 as i64).await {
        return reply_ephemeral(bot, message, "只有本群管理員可以開啟設定。").await;
    }
    let Some(service) = runtime.miniapp.get() else {
        return reply_ephemeral(bot, message, "設定面板尚未開放，請先使用 /module。").await;
    };
    match service
        .launch_link(user.id.0 as i64, message.chat.id.0)
        .await
    {
        Ok(link) => {
            bot.send_message(
                message.chat.id,
                "群組設定\n此連結只限你使用，5 分鐘內有效。",
            )
            .reply_markup(InlineKeyboardMarkup::new(vec![vec![
                InlineKeyboardButton::url("開啟設定", link),
            ]]))
            .await?;
        }
        Err(_) => {
            reply_ephemeral(bot, message, "暫時無法開啟設定，請稍後再試。").await?;
        }
    }
    Ok(())
}
