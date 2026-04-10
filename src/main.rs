use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, env, path::PathBuf, sync::Arc};
use teloxide::{prelude::*, types::{CallbackQuery, ChatId, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Message, MessageId, ParseMode, UserId}};
use url::Url;
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Clone)]
struct Config {
    bot_token: String,
    log_channel_id: i64,
    report_channel_id: i64,
    test_group_id: Option<i64>,
    maintainer_ids: Vec<i64>,
    data_dir: PathBuf,
    sqlite_path: PathBuf,
    spam_threshold: f64,
}

impl Config {
    fn from_env() -> Result<Self> {
        let bot_token = env::var("BOT_TOKEN").context("BOT_TOKEN is required")?;
        let log_channel_id = env::var("LOG_CHANNEL_ID")
            .context("LOG_CHANNEL_ID is required")?
            .parse()?;
        let report_channel_id = env::var("REPORT_CHANNEL_ID")
            .context("REPORT_CHANNEL_ID is required")?
            .parse()?;
        let test_group_id = env::var("TEST_GROUP_ID").ok().and_then(|v| v.parse::<i64>().ok());
        let maintainer_ids = env::var("MAINTAINER_IDS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|v| v.trim().parse::<i64>().ok())
            .collect::<Vec<_>>();
        let data_dir = env::var("DATA_DIR").unwrap_or_else(|_| "data".to_string());
        let sqlite_path = env::var("SQLITE_PATH").unwrap_or_else(|_| format!("{data_dir}/bot.db"));
        let spam_threshold = env::var("SPAM_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.85);
        Ok(Self {
            bot_token,
            log_channel_id,
            report_channel_id,
            test_group_id,
            maintainer_ids,
            data_dir: PathBuf::from(data_dir),
            sqlite_path: PathBuf::from(sqlite_path),
            spam_threshold,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ActionKind {
    AutoDelete,
    AutoBan,
    SpamBan,
    Mute,
    Kick,
    PendingReport,
    ReportApproved,
    ReportRejected,
}

impl ActionKind {
    fn as_str(&self) -> &'static str {
        match self {
            ActionKind::AutoDelete => "auto_delete",
            ActionKind::AutoBan => "auto_ban",
            ActionKind::SpamBan => "spam_ban",
            ActionKind::Mute => "mute",
            ActionKind::Kick => "kick",
            ActionKind::PendingReport => "pending_report",
            ActionKind::ReportApproved => "report_approved",
            ActionKind::ReportRejected => "report_rejected",
        }
    }

    fn from_str(value: &str) -> Self {
        match value {
            "auto_delete" => ActionKind::AutoDelete,
            "auto_ban" => ActionKind::AutoBan,
            "spam_ban" => ActionKind::SpamBan,
            "mute" => ActionKind::Mute,
            "kick" => ActionKind::Kick,
            "pending_report" => ActionKind::PendingReport,
            "report_approved" => ActionKind::ReportApproved,
            "report_rejected" => ActionKind::ReportRejected,
            _ => ActionKind::AutoBan,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CaseRecord {
    id: String,
    action: ActionKind,
    chat_id: i64,
    target_user_id: i64,
    target_name: String,
    actor_user_id: Option<i64>,
    actor_name: Option<String>,
    source_message_id: Option<i32>,
    evidence_text: String,
    model_score: Option<f64>,
    status: String,
    log_message_id: Option<i32>,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ModelState {
    spam_docs: u64,
    ham_docs: u64,
    spam_tokens: HashMap<String, u64>,
    ham_tokens: HashMap<String, u64>,
}

struct Runtime {
    config: Config,
    sqlite_path: PathBuf,
    project_chat: Mutex<Option<i64>>,
    model: Mutex<ModelState>,
    mass_train_buffer: Mutex<HashMap<i64, Vec<String>>>,
    mass_train_mode: Mutex<HashMap<i64, String>>,
}

impl Runtime {
    async fn load(config: Config) -> Result<Self> {
        tokio::fs::create_dir_all(&config.data_dir).await.ok();
        let sqlite_path = config.sqlite_path.clone();
        let conn = Connection::open(&sqlite_path)?;
        Self::init_db(&conn)?;
        let model = Self::load_model(&conn)?;
        let project_chat = Self::load_project_chat(&conn)?;
        Ok(Self { config, sqlite_path, project_chat: Mutex::new(project_chat), model: Mutex::new(model), mass_train_buffer: Mutex::new(HashMap::new()), mass_train_mode: Mutex::new(HashMap::new()) })
    }

    fn open_conn(&self) -> Result<Connection> {
        let conn = Connection::open(&self.sqlite_path)?;
        Self::init_db(&conn)?;
        Ok(conn)
    }

    fn init_db(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            r#"
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS cases (
                id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                target_user_id INTEGER NOT NULL,
                target_name TEXT NOT NULL,
                actor_user_id INTEGER,
                actor_name TEXT,
                source_message_id INTEGER,
                evidence_text TEXT NOT NULL,
                model_score REAL,
                status TEXT NOT NULL,
                log_message_id INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS training_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                text TEXT NOT NULL,
                case_id TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS token_counts (
                token TEXT NOT NULL,
                label TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (token, label)
            );
            CREATE TABLE IF NOT EXISTS model_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            "#,
        )?;
        Ok(())
    }

    fn load_model(conn: &Connection) -> Result<ModelState> {
        let mut model = ModelState::default();
        let mut stmt = conn.prepare("SELECT key, value FROM model_meta")?;
        let rows = stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
        for row in rows {
            let (key, value) = row?;
            match key.as_str() {
                "spam_docs" => model.spam_docs = value.parse().unwrap_or(0),
                "ham_docs" => model.ham_docs = value.parse().unwrap_or(0),
                _ => {}
            }
        }

        let mut stmt = conn.prepare("SELECT token, label, count FROM token_counts")?;
        let rows = stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, u64>(2)?)))?;
        for row in rows {
            let (token, label, count) = row?;
            match label.as_str() {
                "spam" => { model.spam_tokens.insert(token, count); }
                "ham" => { model.ham_tokens.insert(token, count); }
                _ => {}
            }
        }

        Ok(model)
    }

    fn load_threshold(conn: &Connection) -> Result<Option<f64>> {
        let mut stmt = conn.prepare("SELECT value FROM model_meta WHERE key = 'spam_threshold'")?;
        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            let value: String = row.get(0)?;
            Ok(value.parse::<f64>().ok())
        } else {
            Ok(None)
        }
    }

    fn load_project_chat(conn: &Connection) -> Result<Option<i64>> {
        let mut stmt = conn.prepare("SELECT value FROM model_meta WHERE key = 'project_chat_id'")?;
        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            let value: String = row.get(0)?;
            Ok(value.parse::<i64>().ok())
        } else {
            Ok(None)
        }
    }

    async fn persist_case(&self, case: &CaseRecord) -> Result<()> {
        let conn = self.open_conn()?;
        conn.execute(
            r#"
            INSERT INTO cases (id, action, chat_id, target_user_id, target_name, actor_user_id, actor_name, source_message_id, evidence_text, model_score, status, log_message_id, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
            ON CONFLICT(id) DO UPDATE SET
              action=excluded.action,
              chat_id=excluded.chat_id,
              target_user_id=excluded.target_user_id,
              target_name=excluded.target_name,
              actor_user_id=excluded.actor_user_id,
              actor_name=excluded.actor_name,
              source_message_id=excluded.source_message_id,
              evidence_text=excluded.evidence_text,
              model_score=excluded.model_score,
              status=excluded.status,
              log_message_id=excluded.log_message_id,
              created_at=excluded.created_at
            "#,
            params![
                case.id,
                case.action.as_str(),
                case.chat_id,
                case.target_user_id,
                case.target_name,
                case.actor_user_id,
                case.actor_name,
                case.source_message_id,
                case.evidence_text,
                case.model_score,
                case.status,
                case.log_message_id,
                case.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    async fn load_case(&self, case_id: &str) -> Result<Option<CaseRecord>> {
        let conn = self.open_conn()?;
        let result = {
            let mut stmt = conn.prepare(
                r#"SELECT id, action, chat_id, target_user_id, target_name, actor_user_id, actor_name, source_message_id, evidence_text, model_score, status, log_message_id, created_at FROM cases WHERE id = ?1"#,
            )?;
            let mut rows = stmt.query(params![case_id])?;
            if let Some(row) = rows.next()? {
                let created_at = DateTime::parse_from_rfc3339(&row.get::<_, String>(12)?)?.with_timezone(&Utc);
                Some(CaseRecord {
                    id: row.get(0)?,
                    action: ActionKind::from_str(&row.get::<_, String>(1)?),
                    chat_id: row.get(2)?,
                    target_user_id: row.get(3)?,
                    target_name: row.get(4)?,
                    actor_user_id: row.get(5)?,
                    actor_name: row.get(6)?,
                    source_message_id: row.get(7)?,
                    evidence_text: row.get(8)?,
                    model_score: row.get(9)?,
                    status: row.get(10)?,
                    log_message_id: row.get(11)?,
                    created_at,
                })
            } else {
                None
            }
        };
        Ok(result)
    }

    async fn insert_training_sample(&self, label: &str, text: &str, case_id: Option<&str>) -> Result<()> {
        let conn = self.open_conn()?;
        conn.execute(
            "INSERT INTO training_samples (label, text, case_id, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![label, text, case_id, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    async fn purge_training_by_case(&self, case_id: &str) -> Result<usize> {
        let conn = self.open_conn()?;
        let affected = conn.execute("DELETE FROM training_samples WHERE case_id = ?1", params![case_id])?;
        Ok(affected)
    }

    async fn rebuild_model(&self) -> Result<ModelState> {
        let conn = self.open_conn()?;
        let rebuilt = {
            let mut rebuilt = ModelState::default();
            let mut stmt = conn.prepare("SELECT label, text FROM training_samples ORDER BY id ASC")?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let label: String = row.get(0)?;
                let text: String = row.get(1)?;
                match label.as_str() {
                    "spam" => {
                        rebuilt.spam_docs += 1;
                        for token in tokenize(&text) {
                            *rebuilt.spam_tokens.entry(token).or_default() += 1;
                        }
                    }
                    "ham" => {
                        rebuilt.ham_docs += 1;
                        for token in tokenize(&text) {
                            *rebuilt.ham_tokens.entry(token).or_default() += 1;
                        }
                    }
                    _ => {}
                }
            }
            rebuilt
        };

        conn.execute("DELETE FROM model_meta", [])?;
        conn.execute("DELETE FROM token_counts", [])?;

        {
            let mut model = self.model.lock().await;
            *model = rebuilt.clone();
        }
        self.update_model_meta().await?;
        Ok(rebuilt)
    }

    async fn update_model_meta(&self) -> Result<()> {
        let model = self.model.lock().await;
        let conn = self.open_conn()?;
        conn.execute(
            "INSERT INTO model_meta (key, value) VALUES ('spam_docs', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![model.spam_docs.to_string()],
        )?;
        conn.execute(
            "INSERT INTO model_meta (key, value) VALUES ('ham_docs', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![model.ham_docs.to_string()],
        )?;
        for (token, count) in &model.spam_tokens {
            conn.execute(
                "INSERT INTO token_counts (token, label, count) VALUES (?1, 'spam', ?2) ON CONFLICT(token, label) DO UPDATE SET count=excluded.count",
                params![token, count.to_string()],
            )?;
        }
        for (token, count) in &model.ham_tokens {
            conn.execute(
                "INSERT INTO token_counts (token, label, count) VALUES (?1, 'ham', ?2) ON CONFLICT(token, label) DO UPDATE SET count=excluded.count",
                params![token, count.to_string()],
            )?;
        }
        Ok(())
    }

    async fn set_threshold(&self, value: f64) -> Result<()> {
        let conn = self.open_conn()?;
        conn.execute(
            "INSERT INTO model_meta (key, value) VALUES ('spam_threshold', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![value.to_string()],
        )?;
        Ok(())
    }

    async fn current_threshold(&self) -> Result<f64> {
        let conn = self.open_conn()?;
        if let Some(value) = Self::load_threshold(&conn)? {
            Ok(value)
        } else {
            Ok(self.config.spam_threshold)
        }
    }

    async fn start_mass_train(&self, user_id: i64) {
        let mut buffer = self.mass_train_buffer.lock().await;
        buffer.insert(user_id, Vec::new());
    }

    async fn set_mass_train_mode(&self, user_id: i64, mode: &str) {
        let mut modes = self.mass_train_mode.lock().await;
        modes.insert(user_id, mode.to_string());
    }

    async fn mass_train_mode(&self, user_id: i64) -> Option<String> {
        let modes = self.mass_train_mode.lock().await;
        modes.get(&user_id).cloned()
    }

    async fn push_mass_train_text(&self, user_id: i64, text: String) {
        let mut buffer = self.mass_train_buffer.lock().await;
        if let Some(list) = buffer.get_mut(&user_id) {
            list.push(text);
        }
    }

    async fn finish_mass_train(&self, user_id: i64) -> Vec<String> {
        let mut buffer = self.mass_train_buffer.lock().await;
        buffer.remove(&user_id).unwrap_or_default()
    }

    async fn clear_mass_train(&self, user_id: i64) {
        let mut buffer = self.mass_train_buffer.lock().await;
        buffer.remove(&user_id);
        let mut modes = self.mass_train_mode.lock().await;
        modes.remove(&user_id);
    }

    async fn set_project_chat(&self, chat_id: i64) {
        if let Ok(conn) = self.open_conn() {
            let _ = conn.execute(
                "INSERT INTO model_meta (key, value) VALUES ('project_chat_id', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![chat_id.to_string()],
            );
        }
        let mut project_chat = self.project_chat.lock().await;
        *project_chat = Some(chat_id);
    }

    async fn project_chat(&self) -> Option<i64> {
        let project_chat = self.project_chat.lock().await;
        *project_chat
    }

    async fn training_stats(&self) -> Result<(u64, u64, u64)> {
        let conn = self.open_conn()?;
        let spam: u64 = conn.query_row("SELECT COUNT(*) FROM training_samples WHERE label='spam'", [], |row| row.get(0))?;
        let ham: u64 = conn.query_row("SELECT COUNT(*) FROM training_samples WHERE label='ham'", [], |row| row.get(0))?;
        let total: u64 = conn.query_row("SELECT COUNT(*) FROM training_samples", [], |row| row.get(0))?;
        Ok((spam, ham, total))
    }

    async fn export_training_data(&self) -> Result<String> {
        let conn = self.open_conn()?;
        let out = {
            let mut stmt = conn.prepare("SELECT label, text, case_id, created_at FROM training_samples ORDER BY id DESC")?;
            let mut rows = stmt.query([])?;
            let mut out = String::new();
            while let Some(row) = rows.next()? {
                let label: String = row.get(0)?;
                let text: String = row.get(1)?;
                let case_id: Option<String> = row.get(2)?;
                let created_at: String = row.get(3)?;
                out.push_str(&format!("[{created_at}] {label} {} {}\n", case_id.unwrap_or_else(|| "-".to_string()), text.replace('\n', " ")));
            }
            out
        };
        Ok(out)
    }

    async fn effective_threshold(&self) -> Result<f64> {
        self.current_threshold().await
    }
}

#[derive(Debug, Clone)]
enum ModerationCommand {
    Start,
    Help,
    MyId,
    MyChat,
    ScoreTest(String),
    SetChat(String),
    Leave(String),
    SpamBan,
    Mute,
    Kick,
    SpamReport,
    CaseLookup(String),
    MlTrainSpam,
    MlCleanSpam,
    MlStats,
    MlThreshold(String),
    MlExport,
    MlPurge(String),
    MlPurgeText(String),
    MlRebuild,
    MlStartMassTrain,
    MlFinishMassTrain,
    MlImport,
    MlStartMassTrainWithMode(String),
    MlDebugParse,
    Unknown,
}

fn parse_command(text: &str) -> ModerationCommand {
    let head = text.split_whitespace().next().unwrap_or("");
    let base = head.split('@').next().unwrap_or(head).to_lowercase();
    match base.as_str() {
        "/spamban" | "/sb" => ModerationCommand::SpamBan,
        "/mute" | "/m" => ModerationCommand::Mute,
        "/kick" | "/k" => ModerationCommand::Kick,
        "/start" => ModerationCommand::Start,
        "/help" => ModerationCommand::Help,
        "/myid" | "/id" => ModerationCommand::MyId,
        "/mychat" => ModerationCommand::MyChat,
        "/spam" | "/report" => ModerationCommand::SpamReport,
        "/case" | "/lookup" => ModerationCommand::CaseLookup(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/ml_score" | "/score" => ModerationCommand::ScoreTest(text.split_whitespace().skip(1).collect::<Vec<_>>().join(" ")),
        "/setchat" => ModerationCommand::SetChat(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/leave" => ModerationCommand::Leave(text.split_whitespace().skip(1).collect::<Vec<_>>().join(" ")),
        "/ml_train_spam" => ModerationCommand::MlTrainSpam,
        "/ml_clean_spam" => ModerationCommand::MlCleanSpam,
        "/ml_stats" => ModerationCommand::MlStats,
        "/ml_threshold" => ModerationCommand::MlThreshold(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/ml_export" => ModerationCommand::MlExport,
        "/ml_purge" => ModerationCommand::MlPurge(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/ml_purge_text" => ModerationCommand::MlPurgeText(text.split_whitespace().skip(1).collect::<Vec<_>>().join(" ")),
        "/ml_rebuild" => ModerationCommand::MlRebuild,
        "/ml_start_mass_train" => ModerationCommand::MlStartMassTrain,
        "/ml_finish_mass_train" => ModerationCommand::MlFinishMassTrain,
        "/import" => ModerationCommand::MlImport,
        "/ml_start_mass_train_smart" => ModerationCommand::MlStartMassTrainWithMode("smart".to_string()),
        "/ml_start_mass_train_plain" => ModerationCommand::MlStartMassTrainWithMode("plain".to_string()),
        "/ml_debug_parse" => ModerationCommand::MlDebugParse,
        _ => ModerationCommand::Unknown,
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.chars().count() > 1)
        .map(|s| s.to_string())
        .collect()
}

fn feature_text(display_name: &str, text: &str) -> String {
    format!("{display_name} {text}")
}

fn extract_smart_spam_text(input: &str) -> Option<String> {
    let text = input.trim();
    if text.is_empty() {
        return None;
    }

    let mut best = None;
    let patterns = [
        "spam 消息",
        "偵測到廣告",
        "Deleted from",
        "Deleted from ",
        "content of",
        "以下来自",
        "以下來自",
        "以下內容",
    ];

    for pat in patterns {
        if let Some(pos) = text.find(pat) {
            let tail = &text[pos + pat.len()..];
            let tail = tail.trim_start_matches([' ', ':', '：', '\n', '\r']);
            if !tail.is_empty() {
                best = Some(tail.to_string());
            }
        }
    }

    if let Some(pos) = text.rfind("\n\n") {
        let tail = text[pos + 2..].trim();
        if !tail.is_empty() {
            best = Some(tail.to_string());
        }
    }

    best.or_else(|| Some(text.to_string()))
}

fn is_smart_log_header(block: &str) -> bool {
    let lower = block.to_lowercase();
    let markers = [
        "spam 消息",
        "自动删除了以下来自",
        "自動删除了以下来自",
        "偵測到廣告",
        "检测到广告",
        "deleted from",
        "score=",
        "已封鎖該用戶",
        "已封锁该用户",
        "type:",
        "uid:",
        "chat:",
        "mid:",
        "joined chat:",
        "name:",
        "title:",
    ];
    markers.iter().any(|m| lower.contains(m))
}

fn looks_like_metadata_line(line: &str) -> bool {
    let lower = line.trim().to_lowercase();
    lower.starts_with("user ")
        || lower.contains(" joined chat:")
        || lower.contains(" name:")
        || lower.contains(" title:")
        || lower.starts_with("chat:")
        || lower.starts_with("mid:")
        || lower.starts_with("uid:")
}

fn prune_metadata_lines(block: &str) -> String {
    let lines = block
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !looks_like_metadata_line(line))
        .collect::<Vec<_>>();
    lines.join("\n").trim().to_string()
}

fn smart_train_payloads(input: &str) -> Vec<String> {
    let normalized = input.replace("\r\n", "\n");
    let mut out = Vec::new();

    let mut expect_body = false;
    for block in normalized.split("\n\n").map(str::trim).filter(|s| !s.is_empty()) {
        if is_smart_log_header(block) {
            expect_body = true;
            continue;
        }

        if expect_body {
            let candidate = prune_metadata_lines(block);
            if !candidate.is_empty() {
                out.push(candidate.to_string());
            }
            expect_body = false;
            continue;
        }

        let candidate = extract_smart_spam_text(&prune_metadata_lines(block))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| block.to_string());

        if !is_smart_log_header(&candidate) && !candidate.trim().is_empty() {
            out.push(candidate.trim().to_string());
        }
    }

    if out.is_empty() {
        if let Some(single) = extract_smart_spam_text(&normalized) {
            let single = single.trim().to_string();
            if !single.is_empty() {
                out.push(single);
            }
        }
    }

    out.dedup();
    out
}

fn import_train_payloads(input: &str) -> Vec<String> {
    let normalized = input.replace("\r\n", "\n");
    let mut collecting = false;
    let mut out = Vec::new();

    for line in normalized.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.contains("已提取並訓練的字串") {
            collecting = true;
            continue;
        }
        if !collecting {
            continue;
        }
        if trimmed.starts_with("批量訓練完成") {
            break;
        }
        if trimmed == "---" {
            continue;
        }
        if trimmed.starts_with("spam:") || trimmed.starts_with("ham:") || trimmed.starts_with("總樣本:") || trimmed.starts_with("有效門檻:") {
            continue;
        }
        out.push(trimmed.to_string());
    }

    out.dedup();
    out
}

fn help_text() -> &'static str {
    "<b>歡迎使用 Spam Protection Bot（SPB）全自動人工智障反廣告項目。</b>\n\n只需要把這個機器人拉進你的群組，並給它管理員權限（至少需要刪除訊息 + 封禁用戶權限），它就會自動開始工作。\n\n<b>機器人主要功能：</b>\n<code>/sb</code> 或 <code>/spamban</code>：回覆訊息使用，封禁並加入黑名單訓練\n<code>/mute</code>：禁言\n<code>/kick</code>：踢出\n\n普通成員可使用 <code>/report</code> 或 <code>/spam</code> 舉報可疑訊息，交由項目組審核\n任何人可輸入 <code>/case &lt;ID&gt;</code> 查詢某次封禁的詳細記錄\n\n<b>注意事項：</b>\n被封禁後想查原因：先發 <code>/id</code> 取得自己的 User ID，然後去日誌頻道 <code>@SpamProtectionLogging</code> 搜尋\n\n項目交流群：https://t.me/SpamProtectionChat\n日誌頻道：https://t.me/SpamProtectionLogging\n\n<b>項目組指令（僅 maintainer 可見）：</b>\n<code>/ml_train_spam</code>、<code>/ml_clean_spam</code>：單筆訓練/洗樣本\n<code>/ml_purge &lt;case_id&gt;</code>、<code>/ml_purge_text &lt;文字片段&gt;</code>：清除誤樣本\n<code>/ml_rebuild</code>：重建模型\n<code>/ml_stats</code>：查看樣本與門檻\n<code>/ml_threshold &lt;值&gt;</code>：調整自動封禁門檻\n<code>/ml_export</code>：匯出訓練資料\n<code>/import</code>：匯入已輸出的訓練列表\n<code>/ml_start_mass_train_smart</code>：貼原始日志，自動抽正文全當 spam\n<code>/ml_start_mass_train_plain</code>：逐條手工標註\n<code>/ml_finish_mass_train</code>：結束批量訓練\n<code>/ml_debug_parse</code>：測試 smart 抽取"
}

fn score_spam(model: &ModelState, display_name: &str, text: &str) -> f64 {
    let tokens = tokenize(&feature_text(display_name, text));
    if tokens.is_empty() {
        return 0.0;
    }

    let spam_total = model.spam_tokens.values().sum::<u64>() as f64 + 1.0;
    let ham_total = model.ham_tokens.values().sum::<u64>() as f64 + 1.0;
    let vocab = (model.spam_tokens.len() + model.ham_tokens.len()).max(1) as f64;
    let prior_spam = (model.spam_docs as f64 + 1.0) / ((model.spam_docs + model.ham_docs) as f64 + 2.0);
    let prior_ham = 1.0 - prior_spam;

    let mut log_spam = prior_spam.ln();
    let mut log_ham = prior_ham.ln();

    for token in tokens {
        let spam = *model.spam_tokens.get(&token).unwrap_or(&0) as f64 + 1.0;
        let ham = *model.ham_tokens.get(&token).unwrap_or(&0) as f64 + 1.0;
        log_spam += (spam / (spam_total + vocab)).ln();
        log_ham += (ham / (ham_total + vocab)).ln();
    }

    let odds = (log_spam - log_ham).exp();
    odds / (1.0 + odds)
}

fn public_log_link(config: &Config, message_id: i32) -> String {
    let id = config.log_channel_id.abs().to_string().trim_start_matches("100").to_string();
    format!("https://t.me/c/{id}/{message_id}")
}

fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn utc8_display(dt: DateTime<Utc>) -> String {
    (dt + chrono::TimeDelta::hours(8)).format("%Y-%m-%d %H:%M:%S UTC+8").to_string()
}

async fn is_maintainer(bot: &Bot, config: &Config, user_id: i64) -> bool {
    if config.maintainer_ids.contains(&user_id) {
        return true;
    }
    let _ = bot;
    false
}

async fn is_group_admin(bot: &Bot, chat_id: ChatId, user_id: i64) -> bool {
    match bot.get_chat_member(chat_id, UserId(user_id as u64)).await {
        Ok(member) => {
            let status = format!("{:?}", member);
            status.contains("Administrator") || status.contains("Owner")
        }
        Err(_) => false,
    }
}

async fn store_case(runtime: &Runtime, case: &CaseRecord) -> Result<()> {
    runtime.persist_case(case).await
}

fn short_user(user: &teloxide::types::User) -> String {
    let mut name = user.first_name.clone();
    if let Some(last) = &user.last_name {
        name.push(' ');
        name.push_str(last);
    }
    if let Some(username) = &user.username {
        format!("{name} (@{username})")
    } else {
        name
    }
}

fn is_special_user(config: &Config, user_id: i64) -> bool {
    config.maintainer_ids.contains(&user_id)
}

fn project_chat_link(chat_id: i64) -> String {
    let id = chat_id.abs().to_string().trim_start_matches("100").to_string();
    format!("https://t.me/c/{id}/1")
}

fn from_name(user: &teloxide::types::User) -> String {
    short_user(user)
}

async fn log_action(bot: &Bot, runtime: &Runtime, case: &CaseRecord) -> ResponseResult<i32> {
    let text = format!(
        "<b>Case</b>: <code>{}</code>\n<b>Action</b>: {:?}\n<b>Chat</b>: <code>{}</code>\n<b>Target</b>: <code>{}</code> {}\n<b>Actor</b>: {}\n<b>Score</b>: {}\n<b>Evidence</b>:\n<blockquote>{}</blockquote>\n<b>At</b>: {}",
        case.id,
        case.action,
        case.chat_id,
        case.target_user_id,
        escape_html(&case.target_name),
        case.actor_user_id.map(|id| id.to_string()).unwrap_or_else(|| "system".to_string()),
        case.model_score.map(|s| format!("{s:.4}")).unwrap_or_else(|| "-".to_string()),
        escape_html(&case.evidence_text),
        utc8_display(case.created_at),
    );
    let sent = bot
        .send_message(ChatId(runtime.config.log_channel_id), text)
        .parse_mode(ParseMode::Html)
        .await?;
    Ok(sent.id.0)
}

async fn log_callback_error(bot: &Bot, runtime: &Runtime, case: &CaseRecord, stage: &str, err: &str) {
    eprintln!("[callback-error] stage={stage} case={} chat={} err={err}", case.id, case.chat_id);
    let text = format!(
        "<b>Callback Error</b>\n<b>Stage</b>: <code>{}</code>\n<b>Case</b>: <code>{}</code>\n<b>Chat</b>: <code>{}</code>\n<b>Err</b>:\n<blockquote>{}</blockquote>",
        escape_html(stage),
        case.id,
        case.chat_id,
        escape_html(err),
    );
    let _ = bot.send_message(ChatId(runtime.config.log_channel_id), text).parse_mode(ParseMode::Html).await;
}

async fn notify_group(bot: &Bot, runtime: &Runtime, case: &CaseRecord, log_message_id: i32, header: &str) -> Result<()> {
    let link = public_log_link(&runtime.config, log_message_id);
    let text = format!(
        "{header}\n\n<b>對象</b>: <code>{}</code> {}\n<b>證據</b>: <a href=\"{}\">查看日誌</a>\n<b>Case</b>: <code>{}</code>",
        case.target_user_id, escape_html(&case.target_name), link, case.id
    );
    bot.send_message(ChatId(case.chat_id), text).parse_mode(ParseMode::Html).await?;
    Ok(())
}

async fn ban_user(bot: &Bot, chat_id: ChatId, user_id: i64) -> Result<()> {
    bot.ban_chat_member(chat_id, UserId(user_id as u64)).await?;
    Ok(())
}

async fn mute_user(bot: &Bot, chat_id: ChatId, user_id: i64) -> Result<()> {
    let permissions = teloxide::types::ChatPermissions::empty();
    bot.restrict_chat_member(chat_id, UserId(user_id as u64), permissions).await?;
    Ok(())
}

async fn kick_user(bot: &Bot, chat_id: ChatId, user_id: i64) -> Result<()> {
    bot.ban_chat_member(chat_id, UserId(user_id as u64)).await?;
    bot.unban_chat_member(chat_id, UserId(user_id as u64)).await?;
    Ok(())
}

async fn train_spam(runtime: &Runtime, display_name: &str, text: &str, case_id: Option<&str>) -> Result<()> {
    {
        let mut model = runtime.model.lock().await;
        model.spam_docs += 1;
        for token in tokenize(&feature_text(display_name, text)) {
            *model.spam_tokens.entry(token).or_default() += 1;
        }
    }
    runtime.insert_training_sample("spam", text, case_id).await?;
    runtime.update_model_meta().await
}

async fn train_ham(runtime: &Runtime, display_name: &str, text: &str, case_id: Option<&str>) -> Result<()> {
    {
        let mut model = runtime.model.lock().await;
        model.ham_docs += 1;
        for token in tokenize(&feature_text(display_name, text)) {
            *model.ham_tokens.entry(token).or_default() += 1;
        }
    }
    runtime.insert_training_sample("ham", text, case_id).await?;
    runtime.update_model_meta().await
}

async fn extract_reply_context(message: &Message) -> Option<(i64, String, i32, String)> {
    let reply = message.reply_to_message()?;
    let user = reply.from.as_ref()?;
    let text = reply.text().or(reply.caption()).unwrap_or("").to_string();
    Some((user.id.0 as i64, short_user(user), reply.id.0, text))
}

async fn handle_command(bot: Bot, runtime: Arc<Runtime>, message: Message) -> ResponseResult<()> {
    let Some(text) = message.text() else { return Ok(()); };
    let cmd = parse_command(text);
    let Some(from) = message.from.as_ref() else { return Ok(()); };
    let from_id = from.id.0 as i64;
    let is_private_maintainer = message.chat.is_private() && runtime.config.maintainer_ids.contains(&from_id);

    if is_private_maintainer && runtime.mass_train_mode(from_id).await.is_some() && message.text().map(|t| !t.trim_start().starts_with('/')).unwrap_or(false) {
        if let Some(text) = message.text() {
            runtime.push_mass_train_text(from_id, text.to_string()).await;
        }
        return Ok(());
    }

    if is_private_maintainer && runtime.mass_train_mode(from_id).await.is_some() && matches!(cmd, ModerationCommand::Unknown) {
        runtime.push_mass_train_text(from_id, text.to_string()).await;
        return Ok(());
    }

    match cmd {
        ModerationCommand::Start | ModerationCommand::Help => {
            bot.send_message(message.chat.id, help_text()).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::MyId => {
            let uid = message.from.as_ref().map(|u| u.id.0.to_string()).unwrap_or_else(|| "unknown".to_string());
            let maintainer = if is_maintainer(&bot, &runtime.config, from_id).await { "yes" } else { "no" };
            bot.send_message(message.chat.id, format!("你的 Telegram ID: <code>{uid}</code>\n是否在 MAINTAINER_IDS: {maintainer}")).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::MyChat => {
            bot.send_message(message.chat.id, format!("這個群的 Chat ID: <code>{}</code>", message.chat.id.0)).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::ScoreTest(text) => {
            if !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只有維護人員可以使用 /ml_score。") .await?;
                return Ok(());
            }
            let text = if text.trim().is_empty() {
                message
                    .reply_to_message()
                    .and_then(|m| m.text().or(m.caption()))
                    .unwrap_or("")
                    .to_string()
            } else {
                text
            };
            if text.trim().is_empty() {
                bot.send_message(message.chat.id, "請在指令後面提供要測試的文本，或回覆一條消息後使用 /ml_score。") .await?;
                return Ok(());
            }
            let user_name = message.from.as_ref().map(short_user).unwrap_or_else(|| "unknown".to_string());
            let model = runtime.model.lock().await;
            let score = score_spam(&model, &user_name, &text);
            drop(model);
            let threshold = runtime.effective_threshold().await.unwrap_or(runtime.config.spam_threshold);
            bot.send_message(message.chat.id, format!("<b>Score</b>: {score:.4}\n<b>Threshold</b>: {threshold:.4}\n<b>Verdict</b>: {}", if score >= threshold { "spam" } else { "ham" })).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::SetChat(chat_id) => {
            if !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只有維護人員可以設定項目交流群。") .await?;
                return Ok(());
            }
            let Some(value) = chat_id.parse::<i64>().ok() else {
                bot.send_message(message.chat.id, "請提供有效的 Chat ID。") .await?;
                return Ok(());
            };
            runtime.set_project_chat(value).await;
            bot.send_message(message.chat.id, format!("已設定項目交流群為 <code>{value}</code>")).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::Leave(reason) => {
            if !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只有維護人員可以使用 /leave。") .await?;
                return Ok(());
            }
            let reason = if reason.trim().is_empty() { "違反使用規則".to_string() } else { reason };
            let project_chat = match runtime.project_chat().await {
                Some(id) => id,
                None => {
                    bot.send_message(message.chat.id, "尚未設定項目交流群，請先使用 /setchat。") .await?;
                    return Ok(());
                }
            };
            let text = format!("已停止為此群提供服務。原因：{}", escape_html(&reason));
            let button = InlineKeyboardMarkup::new(vec![vec![InlineKeyboardButton::url("前往項目交流群查詢", Url::parse(&project_chat_link(project_chat)).unwrap())]]);
            let _ = bot.send_message(message.chat.id, text).parse_mode(ParseMode::Html).reply_markup(button).await;
            let _ = bot.leave_chat(message.chat.id).await;
            bot.send_message(message.chat.id, "已退出此群。") .await?;
        }
        ModerationCommand::SpamBan | ModerationCommand::Mute | ModerationCommand::Kick => {
            let Some((target_id, target_name, source_id, evidence_text)) = extract_reply_context(&message).await else {
                bot.send_message(message.chat.id, "請回覆一條訊息後再使用此指令。").await?;
                return Ok(());
            };

            if !is_group_admin(&bot, message.chat.id, from_id).await {
                bot.send_message(message.chat.id, "只有群組管理員可以執行此指令。").await?;
                return Ok(());
            }

            if is_group_admin(&bot, message.chat.id, target_id).await || is_special_user(&runtime.config, target_id) {
                bot.send_message(message.chat.id, "不能對群組管理員或項目維護人員執行此指令。").await?;
                return Ok(());
            }

            let action = match cmd {
                ModerationCommand::SpamBan => ActionKind::SpamBan,
                ModerationCommand::Mute => ActionKind::Mute,
                ModerationCommand::Kick => ActionKind::Kick,
                _ => unreachable!(),
            };
            let case_id = Uuid::new_v4().to_string();
            let mut case = CaseRecord {
                id: case_id.clone(),
                action: action.clone(),
                chat_id: message.chat.id.0,
                target_user_id: target_id,
                target_name: target_name.clone(),
                actor_user_id: Some(from_id),
                actor_name: Some(short_user(from)),
                source_message_id: Some(source_id),
                evidence_text: evidence_text.clone(),
                model_score: None,
                status: "done".to_string(),
                log_message_id: None,
                created_at: Utc::now(),
            };

            match action {
                ActionKind::SpamBan => {
                    let _ = bot.delete_message(message.chat.id, MessageId(source_id)).await;
                    ban_user(&bot, message.chat.id, target_id).await.ok();
                    train_spam(&runtime, &target_name, &evidence_text, Some(&case_id)).await.ok();
                }
                ActionKind::Mute => {
                    mute_user(&bot, message.chat.id, target_id).await.ok();
                }
                ActionKind::Kick => {
                    kick_user(&bot, message.chat.id, target_id).await.ok();
                }
                _ => {}
            }

            let log_message_id = log_action(&bot, &runtime, &case).await.unwrap_or_default();
            case.log_message_id = Some(log_message_id);
            store_case(&runtime, &case).await.ok();
            notify_group(&bot, &runtime, &case, log_message_id, "<b>已執行管理操作</b>").await.ok();
        }
        ModerationCommand::SpamReport => {
            let Some((target_id, target_name, source_id, evidence_text)) = extract_reply_context(&message).await else {
                bot.send_message(message.chat.id, "請回覆一條疑似 spam 的訊息。").await?;
                return Ok(());
            };

            let case_id = Uuid::new_v4().to_string();
            let case = CaseRecord {
                id: case_id.clone(),
                action: ActionKind::PendingReport,
                chat_id: message.chat.id.0,
                target_user_id: target_id,
                target_name: target_name.clone(),
                actor_user_id: Some(from_id),
                actor_name: Some(short_user(from)),
                source_message_id: Some(source_id),
                evidence_text: evidence_text.clone(),
                model_score: None,
                status: "pending_review".to_string(),
                log_message_id: None,
                created_at: Utc::now(),
            };

            let keyboard = InlineKeyboardMarkup::new(vec![vec![
                InlineKeyboardButton::callback("受理並封禁", format!("review:approve:{case_id}")),
                InlineKeyboardButton::callback("拒絕並洗模型", format!("review:reject:{case_id}")),
            ]]);

            let text = format!(
                "<b>新的 /spam 申請</b>\n\n<b>對象</b>: {} ({})\n<b>發起人</b>: {}\n<b>內容</b>: <blockquote>{}</blockquote>\n<b>Case</b>: <code>{}</code>",
                target_name,
                target_id,
                short_user(from),
                evidence_text,
                case_id
            );

            let sent = bot
                .send_message(ChatId(runtime.config.report_channel_id), text)
                .parse_mode(ParseMode::Html)
                .reply_markup(keyboard)
                .await?;

            let trace = format!(
                "<b>Report Assigned</b>\n<b>Case</b>: <code>{}</code>\n<b>Handled by</b>: <code>{}</code>\n<b>Source</b>: <code>{}</code>",
                case_id,
                short_user(from),
                short_user(from)
            );
            let _ = bot.send_message(ChatId(runtime.config.report_channel_id), trace).parse_mode(ParseMode::Html).await;

            let mut stored = case.clone();
            stored.log_message_id = Some(sent.id.0);
            stored.status = "pending_review".to_string();
            store_case(&runtime, &stored).await.ok();

            bot.send_message(message.chat.id, "已送交舉報處理頻道審核。")
                .await?;
        }
        ModerationCommand::CaseLookup(case_id) => {
            match runtime.load_case(&case_id).await {
                Ok(Some(case)) => {
                    let link = case.log_message_id.map(|id| public_log_link(&runtime.config, id)).unwrap_or_else(|| "-".to_string());
                    let text = format!(
                        "<b>Case</b>: <code>{}</code>\n<b>Status</b>: {}\n<b>Action</b>: {:?}\n<b>Target</b>: {} ({})\n<b>Evidence</b>: <blockquote>{}</blockquote>\n<b>Log</b>: {}",
                        case.id, case.status, case.action, case.target_name, case.target_user_id, case.evidence_text, link
                    );
                    bot.send_message(message.chat.id, text).parse_mode(ParseMode::Html).await?;
                }
                _ => {
                    bot.send_message(message.chat.id, "找不到該 Case。") .await?;
                }
            }
        }
        ModerationCommand::MlTrainSpam | ModerationCommand::MlCleanSpam => {
            if !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只有項目維護組可以使用此指令。").await?;
                return Ok(());
            }
            let Some(text) = message.reply_to_message().and_then(|m| m.text().or(m.caption())) else {
                bot.send_message(message.chat.id, "請回覆一條訊息來訓練或清洗模型。").await?;
                return Ok(());
            };
            match cmd {
                ModerationCommand::MlTrainSpam => {
                    train_spam(&runtime, &from_name(from), text, None).await.ok();
                    bot.send_message(message.chat.id, "已將該樣本寫入 spam 模型。") .await?;
                }
                ModerationCommand::MlCleanSpam => {
                    train_ham(&runtime, &from_name(from), text, None).await.ok();
                    bot.send_message(message.chat.id, "已將該樣本寫入 ham/clean 模型。") .await?;
                }
                _ => {}
            }
        }
        ModerationCommand::MlPurge(case_id) => {
            if !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只有項目維護組可以使用此指令。").await?;
                return Ok(());
            }
            let removed = runtime.purge_training_by_case(&case_id).await.unwrap_or(0);
            let _ = runtime.rebuild_model().await;
            bot.send_message(message.chat.id, format!("已刪除 {removed} 筆訓練樣本，並重建模型。")) .await?;
        }
        ModerationCommand::MlPurgeText(target) => {
            if !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只有項目維護組可以使用此指令。").await?;
                return Ok(());
            }
            let payload = target.trim();
            if payload.is_empty() {
                bot.send_message(message.chat.id, "請提供要清除的原文片段。").await?;
                return Ok(());
            }
            match runtime.open_conn() {
                Ok(conn) => {
                    let removed = conn.execute(
                        "DELETE FROM training_samples WHERE text LIKE ?1 OR text LIKE ?2",
                        params![format!("%{payload}%"), payload],
                    ).unwrap_or(0);
                    let _ = runtime.rebuild_model().await;
                    bot.send_message(message.chat.id, format!("已依文字清除 {removed} 筆訓練樣本，並重建模型。")) .await?;
                }
                Err(err) => {
                    bot.send_message(message.chat.id, format!("清除失敗：{err}")).await?;
                }
            }
        }
        ModerationCommand::MlRebuild => {
            if !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只有項目維護組可以使用此指令。").await?;
                return Ok(());
            }
            let rebuilt = runtime.rebuild_model().await.unwrap_or_default();
            bot.send_message(message.chat.id, format!("已重建模型，spam_docs={} ham_docs={}", rebuilt.spam_docs, rebuilt.ham_docs)).await?;
        }
        ModerationCommand::MlStats => {
            if !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只有項目維護組可以使用此指令。").await?;
                return Ok(());
            }
            let (spam, ham, total) = runtime.training_stats().await.unwrap_or((0, 0, 0));
            let threshold = runtime.effective_threshold().await.unwrap_or(runtime.config.spam_threshold);
            let text = format!("<b>模型統計</b>\nspam: {spam}\nham: {ham}\n總樣本: {total}\n有效門檻: {threshold:.2}");
            bot.send_message(message.chat.id, text).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::MlThreshold(value) => {
            if !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只有項目維護組可以使用此指令。").await?;
                return Ok(());
            }
            if let Ok(threshold) = value.parse::<f64>() {
                let clamped = threshold.clamp(0.50, 0.99);
                runtime.set_threshold(clamped).await.ok();
                bot.send_message(message.chat.id, format!("已保存門檻: {clamped:.2}")).await?;
            } else {
                bot.send_message(message.chat.id, "請提供 0.50 到 0.99 的數值。").await?;
            }
        }
        ModerationCommand::MlExport => {
            if !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只有項目維護組可以使用此指令。").await?;
                return Ok(());
            }
            let export = runtime.export_training_data().await.unwrap_or_default();
            if export.trim().is_empty() {
                bot.send_message(message.chat.id, "沒有可匯出的訓練資料。").await?;
            } else {
                let filename = format!("training-export-{}.txt", Utc::now().format("%Y%m%d-%H%M%S"));
                bot.send_message(message.chat.id, "正在匯出訓練資料，請稍候...").await?;
                bot.send_document(message.chat.id, InputFile::memory(export.into_bytes()).file_name(filename)).await?;
            }
        }
        ModerationCommand::MlImport => {
            if !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只有維護人員可以使用 /import。") .await?;
                return Ok(());
            }
            let Some(text) = message.reply_to_message().and_then(|m| m.text().or(m.caption())) else {
                bot.send_message(message.chat.id, "請回覆一段匯出列表或輸出結果。") .await?;
                return Ok(());
            };
            let payloads = import_train_payloads(text);
            if payloads.is_empty() {
                bot.send_message(message.chat.id, "沒有找到可匯入的訓練字串。") .await?;
                return Ok(());
            }
            let mut count = 0usize;
            let mut debug = Vec::new();
            for payload in payloads {
                debug.push(payload.clone());
                train_spam(&runtime, &from_name(from), &payload, None).await.ok();
                count += 1;
            }
            bot.send_message(message.chat.id, format!("已匯入並訓練 {count} 筆。\n\n匯入字串：\n{}", debug.join("\n---\n"))).await?;
        }
        ModerationCommand::MlStartMassTrain => {
            if !message.chat.is_private() || !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只允許維護者在私訊中啟動批量訓練。") .await?;
                return Ok(());
            }
            runtime.start_mass_train(from_id).await;
            runtime.set_mass_train_mode(from_id, "smart").await;
            bot.send_message(message.chat.id, "已啟動批量訓練。接下來你在這個私訊中傳送的純文本訊息會被收集；完成後使用 /ml_finish_mass_train。")
                .await?;
        }
        ModerationCommand::MlStartMassTrainWithMode(mode) => {
            if !message.chat.is_private() || !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只允許維護者在私訊中啟動批量訓練。") .await?;
                return Ok(());
            }
            runtime.start_mass_train(from_id).await;
            runtime.set_mass_train_mode(from_id, &mode).await;
            bot.send_message(message.chat.id, format!("已啟動批量訓練，模式: {mode}。"))
                .await?;
        }
        ModerationCommand::MlDebugParse => {
            if !message.chat.is_private() || !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只允許維護者在私訊中使用 /ml_debug_parse。") .await?;
                return Ok(());
            }
            let Some(text) = message.reply_to_message().and_then(|m| m.text().or(m.caption()).map(|s| s.to_string())) else {
                bot.send_message(message.chat.id, "請回覆一段日誌或訊息內容。") .await?;
                return Ok(());
            };
            let extracted = smart_train_payloads(&text);
            let body = if extracted.is_empty() {
                "<無法提取>".to_string()
            } else {
                extracted.into_iter().map(|s| escape_html(&s)).collect::<Vec<_>>().join("\n---\n")
            };
            bot.send_message(message.chat.id, format!("提取結果：\n<blockquote>{}</blockquote>", body)).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::MlFinishMassTrain => {
            if !message.chat.is_private() || !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只允許維護者在私訊中結束批量訓練。") .await?;
                return Ok(());
            }
            let mode = runtime.mass_train_mode(from_id).await.unwrap_or_else(|| "smart".to_string());
            let samples = runtime.finish_mass_train(from_id).await;
            let mut extracted_debug = Vec::new();
            let mut spam_count = 0usize;
            let mut ham_count = 0usize;
            for sample in samples {
                if sample.trim().is_empty() { continue; }
                if mode == "smart" {
                    let payloads = smart_train_payloads(&sample);
                    if payloads.is_empty() {
                        continue;
                    }
                    for payload in payloads {
                        extracted_debug.push(payload.clone());
                        train_spam(&runtime, "mass-train", &payload, None).await.ok();
                        spam_count += 1;
                    }
                } else {
                    let payload = sample.clone();
                    extracted_debug.push(payload.clone());
                    if sample.starts_with("-") || sample.starts_with("ham:") {
                        train_ham(&runtime, "mass-train", &payload, None).await.ok();
                        ham_count += 1;
                    } else {
                        train_spam(&runtime, "mass-train", &payload, None).await.ok();
                        spam_count += 1;
                    }
                }
            }
            let debug = if extracted_debug.is_empty() { "無可提取樣本".to_string() } else { extracted_debug.join("\n---\n") };
            bot.send_message(message.chat.id, format!("批量訓練完成。spam: {spam_count}, ham: {ham_count}\n\n已提取並訓練的字串：\n{debug}")).await?;
            runtime.clear_mass_train(from_id).await;
        }
        ModerationCommand::Unknown => {
            if message.chat.is_private() {
                bot.send_message(message.chat.id, help_text()).parse_mode(ParseMode::Html).await?;
            }
        }
    }
    Ok(())
}

async fn handle_callback(bot: Bot, runtime: Arc<Runtime>, q: CallbackQuery) -> ResponseResult<()> {
    let Some(data) = q.data.clone() else { return Ok(()); };
    let from = q.from.clone();
    let from_id = from.id.0 as i64;

    eprintln!("[callback] from={} data={}", from_id, data);

    let mut parts = data.split(':');
    let kind = parts.next().unwrap_or("");
    let decision = parts.next().unwrap_or("");
    let case_id = parts.next().unwrap_or("");
    if kind != "review" || case_id.is_empty() {
        return Ok(());
    }

    if q.message.as_ref().map(|m| m.chat().id.0 != runtime.config.report_channel_id).unwrap_or(true) {
        bot.answer_callback_query(q.id).text("此按鈕只能在舉報處理頻道使用").await?;
        return Ok(());
    }

    let case = match runtime.load_case(case_id).await {
        Ok(case) => case,
        Err(_) => {
            bot.answer_callback_query(q.id).text("讀取 Case 失敗").await?;
            return Ok(());
        }
    };

    let Some(case) = case else {
        bot.answer_callback_query(q.id).text("Case 不存在或已處理").await?;
        return Ok(());
    };

    let Some(message) = q.message.as_ref() else {
        bot.answer_callback_query(q.id).text("找不到原始訊息").await?;
        return Ok(());
    };

    match decision {
        "approve" => {
            if let Err(err) = ban_user(&bot, ChatId(case.chat_id), case.target_user_id).await {
                log_callback_error(&bot, &runtime, &case, "ban", &err.to_string()).await;
                bot.answer_callback_query(q.id).text("封禁失敗").await?;
                return Ok(());
            }
            if let Some(source_id) = case.source_message_id {
                if let Err(err) = bot.delete_message(ChatId(case.chat_id), MessageId(source_id)).await {
                    log_callback_error(&bot, &runtime, &case, "delete_message", &err.to_string()).await;
                }
            }
            if let Err(err) = train_spam(&runtime, &case.target_name, &case.evidence_text, Some(&case.id)).await {
                log_callback_error(&bot, &runtime, &case, "train_spam", &err.to_string()).await;
            }
            let mut updated = case.clone();
            updated.action = ActionKind::ReportApproved;
            updated.status = "approved_and_banned".to_string();
            updated.actor_user_id = Some(from_id);
            updated.actor_name = Some(short_user(&from));
            if let Err(err) = store_case(&runtime, &updated).await {
                log_callback_error(&bot, &runtime, &case, "store_case", &err.to_string()).await;
            }
            let body = format!(
                "<b>新的 /spam 申請</b>\n\n<b>對象</b>: {} ({})\n<b>發起人</b>: {}\n<b>內容</b>: <blockquote>{}</blockquote>\n<b>Case</b>: <code>{}</code>\n<b>狀態</b>: 已受理並封禁\n<b>處理者</b>: <code>{}</code>",
                escape_html(&case.target_name),
                case.target_user_id,
                escape_html(&short_user(&from)),
                escape_html(&case.evidence_text),
                case.id,
                from_id
            );
            let _ = bot.edit_message_text(message.chat().id, message.id(), body).parse_mode(ParseMode::Html).await;
            let _ = bot.edit_message_reply_markup(message.chat().id, message.id()).await;
            bot.answer_callback_query(q.id).text("已受理並封禁").await?;
        }
        "reject" => {
            if let Err(err) = train_ham(&runtime, &case.target_name, &case.evidence_text, Some(&case.id)).await {
                log_callback_error(&bot, &runtime, &case, "train_ham", &err.to_string()).await;
            }
            let mut updated = case.clone();
            updated.action = ActionKind::ReportRejected;
            updated.status = "rejected_and_cleaned".to_string();
            updated.actor_user_id = Some(from_id);
            updated.actor_name = Some(short_user(&from));
            if let Err(err) = store_case(&runtime, &updated).await {
                log_callback_error(&bot, &runtime, &case, "store_case", &err.to_string()).await;
            }
            let body = format!(
                "<b>新的 /spam 申請</b>\n\n<b>對象</b>: {} ({})\n<b>發起人</b>: {}\n<b>內容</b>: <blockquote>{}</blockquote>\n<b>Case</b>: <code>{}</code>\n<b>狀態</b>: 已拒絕受理\n<b>處理者</b>: <code>{}</code>",
                escape_html(&case.target_name),
                case.target_user_id,
                escape_html(&short_user(&from)),
                escape_html(&case.evidence_text),
                case.id,
                from_id
            );
            let _ = bot.edit_message_text(message.chat().id, message.id(), body).parse_mode(ParseMode::Html).await;
            let _ = bot.edit_message_reply_markup(message.chat().id, message.id()).await;
            bot.answer_callback_query(q.id).text("已拒絕受理").await?;
        }
        _ => {}
    }

    Ok(())
}

async fn auto_moderate(bot: Bot, runtime: Arc<Runtime>, message: Message) -> ResponseResult<()> {
    let Some(user) = message.from.as_ref() else { return Ok(()); };
    if user.is_bot {
        return Ok(());
    }
    if is_group_admin(&bot, message.chat.id, user.id.0 as i64).await || is_special_user(&runtime.config, user.id.0 as i64) {
        return Ok(());
    }
    let Some(text) = message.text().or(message.caption()) else { return Ok(()); };

    let model = runtime.model.lock().await;
    let display_name = short_user(user);
    let score = score_spam(&model, &display_name, text);
    drop(model);

    let threshold = runtime.effective_threshold().await.unwrap_or(runtime.config.spam_threshold);
    if score < threshold {
        return Ok(());
    }

    let case_id = Uuid::new_v4().to_string();
    let mut case = CaseRecord {
        id: case_id,
        action: ActionKind::AutoBan,
        chat_id: message.chat.id.0,
        target_user_id: user.id.0 as i64,
        target_name: short_user(user),
        actor_user_id: None,
        actor_name: None,
        source_message_id: Some(message.id.0),
        evidence_text: text.to_string(),
        model_score: Some(score),
        status: "auto_banned".to_string(),
        log_message_id: None,
        created_at: Utc::now(),
    };

    let _ = bot.delete_message(message.chat.id, message.id).await;
    let _ = ban_user(&bot, message.chat.id, user.id.0 as i64).await;
    let log_message_id = log_action(&bot, &runtime, &case).await.unwrap_or_default();
    case.log_message_id = Some(log_message_id);
    store_case(&runtime, &case).await.ok();
    notify_group(&bot, &runtime, &case, log_message_id, "<b>自動機器學習封禁</b>").await.ok();
    Ok(())
}

async fn score_only(bot: &Bot, runtime: &Runtime, message: &Message) -> ResponseResult<()> {
    let Some(user) = message.from.as_ref() else { return Ok(()); };
    if user.is_bot {
        return Ok(());
    }
    let Some(text) = message.text().or(message.caption()) else { return Ok(()); };
    let model = runtime.model.lock().await;
    let display_name = short_user(user);
    let score = score_spam(&model, &display_name, text);
    drop(model);
    let threshold = runtime.effective_threshold().await.unwrap_or(runtime.config.spam_threshold);
    let verdict = if score >= threshold { "spam" } else { "ham" };
    let reply = format!("<b>Score</b>: {score:.4}\n<b>Threshold</b>: {threshold:.4}\n<b>Verdict</b>: {verdict}");
    bot.send_message(message.chat.id, reply).parse_mode(ParseMode::Html).await?;
    Ok(())
}

async fn ensure_bot_can_moderate(bot: &Bot, _runtime: &Runtime, chat_id: ChatId) -> ResponseResult<bool> {
    let me = bot.get_me().await?;
    let member = match bot.get_chat_member(chat_id, me.id).await {
        Ok(m) => m,
        Err(_) => {
            let _ = bot.send_message(chat_id, "機器人無法檢查權限，將退出此群。請確認管理員權限後再邀請。" ).await;
            let _ = bot.leave_chat(chat_id).await;
            return Ok(false);
        }
    };
    let status = format!("{:?}", member);
    let allowed = status.contains("Administrator") || status.contains("Owner");
    if !allowed {
        let _ = bot.send_message(chat_id, "機器人缺乏足夠的管理員權限，將退出此群。請確認至少具備刪訊息、封禁、解除封禁、禁言、踢出權限。" ).await;
        let _ = bot.leave_chat(chat_id).await;
    }
    Ok(allowed)
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::from_env()?;
    let bot = Bot::new(config.bot_token.clone());
    let runtime = Arc::new(Runtime::load(config).await?);

    let message_handler = Update::filter_message().endpoint({
        let runtime = runtime.clone();
        move |bot: Bot, message: Message| {
            let runtime = runtime.clone();
            async move {
                if let Some(text) = message.text() {
                    if text.trim_start().starts_with('/') {
                        if !matches!(parse_command(text), ModerationCommand::Unknown) {
                            return handle_command(bot, runtime, message).await;
                        }
                        return Ok(());
                    }
                    if runtime.config.test_group_id == Some(message.chat.id.0) {
                        return score_only(&bot, &runtime, &message).await;
                    }
                    if message.chat.is_private() {
                        return handle_command(bot, runtime, message).await;
                    }
                    if !ensure_bot_can_moderate(&bot, &runtime, message.chat.id).await? {
                        return Ok(());
                    }
                    if matches!(parse_command(text), ModerationCommand::Unknown) {
                        auto_moderate(bot, runtime, message).await?;
                        return Ok(());
                    }
                } else {
                    if runtime.config.test_group_id == Some(message.chat.id.0) {
                        return score_only(&bot, &runtime, &message).await;
                    }
                    if message.chat.is_private() {
                        return handle_command(bot, runtime, message).await;
                    }
                    if !ensure_bot_can_moderate(&bot, &runtime, message.chat.id).await? {
                        return Ok(());
                    }
                    auto_moderate(bot, runtime, message).await?;
                    return Ok(());
                }
                handle_command(bot, runtime, message).await
            }
        }
    });

    let callback_handler = Update::filter_callback_query().endpoint({
        let runtime = runtime.clone();
        move |bot: Bot, q: CallbackQuery| {
            let runtime = runtime.clone();
            async move { handle_callback(bot, runtime, q).await }
        }
    });

    let handler = dptree::entry().branch(message_handler).branch(callback_handler);

    Dispatcher::builder(bot, handler)
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;

    Ok(())
}
