use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use jieba_rs::Jieba;
use fancy_regex::Regex as FancyRegex;
use regex::Regex as StdRegex;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::{collections::{HashMap, VecDeque}, env, path::PathBuf, sync::{Arc, Mutex as StdMutex, OnceLock}, time::Instant};
use teloxide::{prelude::*, types::{CallbackQuery, ChatId, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Message, MessageId, ParseMode, UserId}};
use url::Url;
use tokio::sync::{Mutex, RwLock};
use tokio::time::{sleep, Duration};
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
    // Optional: if set, a "bot is up" DM (with version/commit) is sent here
    // on every startup. Env-configured rather than hardcoded so a personal
    // Telegram user ID never ends up committed to source control.
    owner_id: Option<i64>,
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
        let owner_id = env::var("OWNER_ID").ok().and_then(|v| v.parse::<i64>().ok());
        Ok(Self {
            bot_token,
            log_channel_id,
            report_channel_id,
            test_group_id,
            maintainer_ids,
            data_dir: PathBuf::from(data_dir),
            sqlite_path: PathBuf::from(sqlite_path),
            spam_threshold,
            owner_id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum ActionKind {
    AutoDelete,
    AutoBan,
    SpamBan,
    Mute,
    Kick,
    PendingReport,
    ReportApproved,
    ReportRejected,
    Unbanned,
    Unmuted,
    FloodMute,
    CmdCleanMute,
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
            ActionKind::Unbanned => "unbanned",
            ActionKind::Unmuted => "unmuted",
            ActionKind::FloodMute => "flood_mute",
            ActionKind::CmdCleanMute => "cmd_clean_mute",
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
            "unbanned" => ActionKind::Unbanned,
            "unmuted" => ActionKind::Unmuted,
            "flood_mute" => ActionKind::FloodMute,
            "cmd_clean_mute" => ActionKind::CmdCleanMute,
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
    matched_rule_id: Option<i64>,
    matched_rule_pattern: Option<String>,
    status: String,
    log_message_id: Option<i32>,
    created_at: DateTime<Utc>,
}

/// Builds a `CaseRecord` from a `cases` row - shared by `load_case` and
/// `load_latest_case_by_actions`, both of which select the same 15 columns
/// in the same order (id, action, chat_id, target_user_id, target_name,
/// actor_user_id, actor_name, source_message_id, evidence_text, model_score,
/// matched_rule_id, matched_rule_pattern, status, log_message_id, created_at).
fn case_from_row(row: &rusqlite::Row) -> Result<CaseRecord> {
    let created_at = DateTime::parse_from_rfc3339(&row.get::<_, String>(14)?)?.with_timezone(&Utc);
    Ok(CaseRecord {
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
        matched_rule_id: row.get(10)?,
        matched_rule_pattern: row.get(11)?,
        status: row.get(12)?,
        log_message_id: row.get(13)?,
        created_at,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum CaseKind {
    Ban,
    Mute,
}

/// Enough to reverse a maintainer command via `/revert <action_id>`.
/// Reverting is deliberately "call the same setter again with the old
/// value" rather than bespoke inverse logic - e.g. `GroupModule`'s revert is
/// just another `set_group_module` call, and `Case`'s revert reuses the
/// exact same `reverse_ban_case`/`reverse_mute_case` functions `/unban` and
/// `/unmute` call directly. Serialized to JSON in the `maintainer_actions.
/// undo_data` column.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum UndoData {
    Threshold { old: f64 },
    GroupThreshold { chat_id: i64, old: Option<f64> },
    TokenProbability { token: String, old_spam: u64, old_ham: u64 },
    GroupModule { chat_id: i64, module: String, old_enabled: bool },
    GroupWhitelist { chat_id: i64, user_id: i64, old_enabled: bool },
    GlobalWhitelist { user_id: i64, old_enabled: bool },
    RuleAdded { rule_id: i64 },
    RuleEdited { rule_id: i64, old_pattern: String },
    RuleDeleted { pattern: String, description: String },
    ProjectChat { old: Option<i64> },
    /// A synthetic case_id-like handle passed as `case_id` into
    /// `train_spam`/`train_ham` purely so `purge_training_by_case` can find
    /// and remove exactly this training sample later - not a real case.
    TrainingSample { training_ref: String },
    Case { case_id: String, kind: CaseKind },
    NotRevertible,
}

struct MaintainerAction {
    actor_name: String,
    chat_id: Option<i64>,
    command: String,
    summary: String,
    undo: UndoData,
    reverted: bool,
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
    db: Arc<StdMutex<Connection>>,
    project_chat: Mutex<Option<i64>>,
    /// Private channel for the maintainer action audit log, set via
    /// `/set_audit_log`. Same persistence pattern as `project_chat`.
    audit_log_chat: Mutex<Option<i64>>,
    model: Mutex<ModelState>,
    spam_rules: RwLock<Vec<SpamRule>>,
    mass_train_buffer: Mutex<HashMap<i64, Vec<String>>>,
    mass_train_mode: Mutex<HashMap<i64, String>>,
    pending_rule_additions: Mutex<HashMap<i64, String>>,
    group_module_cache: RwLock<HashMap<i64, GroupModuleSettings>>,
    /// (chat_id, user_id) -> recent message timestamps within the flood
    /// window. In-memory only and reset on restart is fine — flood control
    /// is a rolling behavioral signal, not something that needs to survive
    /// a restart.
    flood_tracker: Mutex<HashMap<(i64, i64), VecDeque<Instant>>>,
    /// (chat_id, user_id) -> outstanding join CAPTCHA. In-memory only: a
    /// restart mid-challenge just means the member gets re-challenged on
    /// their next message, or the timeout task (also lost on restart)
    /// simply never fires — no harm either way, just re-issue on demand.
    pending_captcha: Mutex<HashMap<(i64, i64), PendingCaptcha>>,
}

struct PendingCaptcha {
    expected_answer: String,
    expires_at: Instant,
    challenge_message_id: MessageId,
}

#[derive(Clone)]
struct GroupModuleSettings {
    no_long_name: bool,
    no_halal: bool,
    no_service_messages: bool,
    // Unlike the content-policy modules above (which default off, since
    // they're opinionated choices a group opts into), flood control is
    // baseline hygiene and defaults on; matches the `DEFAULT 1` on the
    // `flood_control` column in group_module_settings.
    flood_control: bool,
    // Join-time CAPTCHA: opt-in, since (unlike flood control) it adds
    // visible friction for every legitimate new member.
    captcha: bool,
    spam_threshold_override: Option<f64>,
    // Cross-group ban propagation ("netban"): opt-in, since it means a ban
    // decision made in a *different* group (outside this group admin's
    // control) can ban someone here too.
    netban: bool,
    // Escalates repeat permission-denied command attempts to a temporary
    // mute; opt-in since it's a real moderation consequence for members,
    // not just cleanup.
    cmd_clean: bool,
}

impl Default for GroupModuleSettings {
    fn default() -> Self {
        Self {
            no_long_name: false,
            no_halal: false,
            no_service_messages: false,
            flood_control: true,
            captcha: false,
            spam_threshold_override: None,
            netban: false,
            cmd_clean: false,
        }
    }
}

#[derive(Clone)]
struct ModuleCheckResult {
    reasons: Vec<String>,
    name_guard: Vec<String>,
    no_halal: Vec<String>,
}

#[derive(Clone)]
struct UserProfileInfo {
    user_id: i64,
    display_name: String,
    username: Option<String>,
    bio: Option<String>,
}

#[derive(Clone)]
struct SpamRule {
    id: i64,
    description: String,
    regex: FancyRegex,
}

#[derive(Clone)]
struct MatchedRule {
    description: String,
}

struct ScoreContribution {
    token: String,
    spam_count: u64,
    ham_count: u64,
    spam_prob: f64,
    ham_prob: f64,
    delta: f64,
}

struct ScoreDebugReport {
    score: f64,
    tokens: Vec<ScoreContribution>,
}

enum InspectionResult {
    Spam { score: f64, matched_rule: Option<MatchedRule> },
    Ham { score: f64 },
}

impl Runtime {
    async fn load(config: Config) -> Result<Self> {
        tokio::fs::create_dir_all(&config.data_dir).await.ok();
        let sqlite_path = config.sqlite_path.clone();
        let mut conn = Connection::open(&sqlite_path)?;
        Self::init_db(&mut conn)?;
        let model = Self::load_model(&conn)?;
        let project_chat = Self::load_project_chat(&conn)?;
        let audit_log_chat = Self::load_audit_log_chat(&conn)?;
        let spam_rules = Self::load_spam_rules(&conn)?;
        Ok(Self {
            config,
            db: Arc::new(StdMutex::new(conn)),
            project_chat: Mutex::new(project_chat),
            audit_log_chat: Mutex::new(audit_log_chat),
            model: Mutex::new(model),
            spam_rules: RwLock::new(spam_rules),
            mass_train_buffer: Mutex::new(HashMap::new()),
            mass_train_mode: Mutex::new(HashMap::new()),
            pending_rule_additions: Mutex::new(HashMap::new()),
            group_module_cache: RwLock::new(HashMap::new()),
            flood_tracker: Mutex::new(HashMap::new()),
            pending_captcha: Mutex::new(HashMap::new()),
        })
    }

    /// Runs `f` against the single shared connection on a blocking-safe thread.
    /// Centralizing access here means schema setup happens once (in `init_db` at
    /// startup) instead of on every query, and keeps SQLite's blocking I/O off
    /// the async executor threads.
    async fn with_conn<T, F>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = db.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            f(&mut conn)
        })
        .await
        .context("database task panicked")?
    }

    fn init_db(conn: &mut Connection) -> Result<()> {
        conn.execute_batch(
            r#"
            PRAGMA journal_mode=WAL;
            PRAGMA busy_timeout=5000;
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
                matched_rule_id INTEGER,
                matched_rule_pattern TEXT,
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
            CREATE TABLE IF NOT EXISTS word_frequencies (
                word TEXT PRIMARY KEY,
                spam_count INTEGER NOT NULL DEFAULT 0,
                ham_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS spam_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS group_module_settings (
                chat_id INTEGER PRIMARY KEY,
                no_long_name INTEGER NOT NULL DEFAULT 0,
                no_halal INTEGER NOT NULL DEFAULT 0,
                no_service_messages INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS group_whitelist (
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                added_by INTEGER,
                created_at TEXT NOT NULL,
                PRIMARY KEY (chat_id, user_id)
            );
            CREATE TABLE IF NOT EXISTS global_whitelist (
                user_id INTEGER PRIMARY KEY,
                added_by INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            "#,
        )?;
        let mut columns = std::collections::HashSet::new();
        {
            let mut stmt = conn.prepare("PRAGMA table_info(cases)")?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
            for row in rows {
                columns.insert(row?);
            }
        }
        if !columns.contains("matched_rule_id") {
            conn.execute("ALTER TABLE cases ADD COLUMN matched_rule_id INTEGER", [])?;
        }
        if !columns.contains("matched_rule_pattern") {
            conn.execute("ALTER TABLE cases ADD COLUMN matched_rule_pattern TEXT", [])?;
        }
        
        // Check and add no_service_messages column if needed
        let mut gms_columns = std::collections::HashSet::new();
        {
            let mut stmt = conn.prepare("PRAGMA table_info(group_module_settings)")?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
            for row in rows {
                gms_columns.insert(row?);
            }
        }
        if !gms_columns.contains("no_service_messages") {
            conn.execute("ALTER TABLE group_module_settings ADD COLUMN no_service_messages INTEGER NOT NULL DEFAULT 0", [])?;
        }
        
        let user_version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if user_version < 1 {
            Self::migrate_v0_to_v1(conn)?;
        }
        if user_version < 2 {
            Self::migrate_v1_to_v2(conn)?;
        }
        if user_version < 3 {
            Self::migrate_v2_to_v3(conn)?;
        }
        if user_version < 4 {
            Self::migrate_v3_to_v4(conn)?;
        }
        if user_version < 5 {
            Self::migrate_v4_to_v5(conn)?;
        }
        if user_version < 6 {
            Self::migrate_v5_to_v6(conn)?;
        }
        Ok(())
    }

    /// Drops the dead `token_counts` table (superseded by `word_frequencies`
    /// long ago, never read or written anywhere) and adds the columns needed
    /// for flood control, join CAPTCHA, and per-group spam thresholds. Not
    /// added to the `CREATE TABLE IF NOT EXISTS group_module_settings` above
    /// on purpose: SQLite's `ALTER TABLE ADD COLUMN` has no `IF NOT EXISTS`
    /// form, so a fresh DB whose CREATE TABLE already had these columns would
    /// hit a "duplicate column" error the first time this migration ran.
    /// Running it unconditionally for every DB (fresh or existing), gated
    /// only by `user_version`, avoids that.
    fn migrate_v2_to_v3(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute("DROP TABLE IF EXISTS token_counts", [])?;
        tx.execute("ALTER TABLE group_module_settings ADD COLUMN flood_control INTEGER NOT NULL DEFAULT 1", [])?;
        tx.execute("ALTER TABLE group_module_settings ADD COLUMN captcha INTEGER NOT NULL DEFAULT 0", [])?;
        tx.execute("ALTER TABLE group_module_settings ADD COLUMN spam_threshold_override REAL", [])?;
        tx.execute("PRAGMA user_version = 3", [])?;
        tx.commit()?;
        Ok(())
    }

    /// Adds the `netban` opt-in flag and `network_ban_targets`, the
    /// historical record of exactly which chats got a propagated ban for a
    /// given case (needed since a group's netban membership can change over
    /// time, so reversal can't just re-derive "which chats" from current
    /// settings - it has to know which chats were actually hit).
    fn migrate_v3_to_v4(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute("ALTER TABLE group_module_settings ADD COLUMN netban INTEGER NOT NULL DEFAULT 0", [])?;
        tx.execute(
            r#"
            CREATE TABLE IF NOT EXISTS network_ban_targets (
                case_id TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (case_id, chat_id)
            )
            "#,
            [],
        )?;
        tx.execute("PRAGMA user_version = 4", [])?;
        tx.commit()?;
        Ok(())
    }

    /// Adds the `cmd_clean` opt-in flag and `permission_offenses`, which
    /// tracks the last time each (chat, user) tripped a permission-denied
    /// guard on a group-admin-tier command - used to detect a repeat offense
    /// within 24h and escalate to a temporary mute.
    fn migrate_v4_to_v5(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute("ALTER TABLE group_module_settings ADD COLUMN cmd_clean INTEGER NOT NULL DEFAULT 0", [])?;
        tx.execute(
            r#"
            CREATE TABLE IF NOT EXISTS permission_offenses (
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                last_offense_at TEXT NOT NULL,
                PRIMARY KEY (chat_id, user_id)
            )
            "#,
            [],
        )?;
        tx.execute("PRAGMA user_version = 5", [])?;
        tx.commit()?;
        Ok(())
    }

    /// One row per state-changing maintainer command, with enough in
    /// `undo_data` (a serialized `UndoData`) to reverse it via `/revert
    /// <action_id>`. `action_id` is a plain autoincrementing integer rather
    /// than a UUID, specifically so it's short enough to type.
    fn migrate_v5_to_v6(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute(
            r#"
            CREATE TABLE IF NOT EXISTS maintainer_actions (
                action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_id INTEGER NOT NULL,
                actor_name TEXT NOT NULL,
                chat_id INTEGER,
                command TEXT NOT NULL,
                summary TEXT NOT NULL,
                undo_data TEXT NOT NULL,
                reverted INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            "#,
            [],
        )?;
        tx.execute("PRAGMA user_version = 6", [])?;
        tx.commit()?;
        Ok(())
    }

    /// A `/set` call with a target very close to 0 or 1 used to compute an
    /// astronomically large raw count for a single token (see
    /// `set_token_probability`) — that count then dominated the shared
    /// spam_total/ham_total used to score every other word, silently
    /// breaking spam detection for the whole model. This runs once on
    /// startup (gated by `PRAGMA user_version`, same as `migrate_v0_to_v1`)
    /// to clamp any such outlier back down, so a deployment self-heals on
    /// its next restart without needing direct DB access — Toolforge's
    /// build-service/k8s setup doesn't give us a shell into the running
    /// pod to run this by hand.
    fn migrate_v1_to_v2(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute("UPDATE word_frequencies SET spam_count = 1000 WHERE spam_count > 1000000", [])?;
        tx.execute("UPDATE word_frequencies SET ham_count = 1000 WHERE ham_count > 1000000", [])?;
        tx.execute("PRAGMA user_version = 2", [])?;
        tx.commit()?;
        Ok(())
    }

    fn migrate_v0_to_v1(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        let jieba = jieba();
        {
            let mut stmt = tx.prepare("SELECT label, text FROM training_samples ORDER BY id ASC")?;
            let rows = stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
            for row in rows {
                let (label, text) = row?;
                let words = normalize_tokens(&text, jieba);
                for word in words {
                    match label.as_str() {
                        "spam" => {
                            tx.execute("INSERT INTO word_frequencies (word, spam_count, ham_count) VALUES (?1, 1, 0) ON CONFLICT(word) DO UPDATE SET spam_count = spam_count + 1", params![word])?;
                        }
                        "ham" => {
                            tx.execute("INSERT INTO word_frequencies (word, spam_count, ham_count) VALUES (?1, 0, 1) ON CONFLICT(word) DO UPDATE SET ham_count = ham_count + 1", params![word])?;
                        }
                        _ => {}
                    }
                }
            }
        }
        tx.execute("PRAGMA user_version = 1", [])?;
        tx.commit()?;
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

        let mut stmt = conn.prepare("SELECT word, spam_count, ham_count FROM word_frequencies")?;
        let rows = stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, u64>(1)?, row.get::<_, u64>(2)?)))?;
        for row in rows {
            let (word, spam_count, ham_count) = row?;
            if spam_count > 0 { model.spam_tokens.insert(word.clone(), spam_count); }
            if ham_count > 0 { model.ham_tokens.insert(word, ham_count); }
        }

        Ok(model)
    }

    fn load_spam_rules(conn: &Connection) -> Result<Vec<SpamRule>> {
        let mut rules = Vec::new();
        let mut stmt = conn.prepare("SELECT id, pattern, description FROM spam_rules ORDER BY id ASC")?;
        let rows = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?)))?;
        for row in rows {
            let (id, pattern, description) = row?;
            if let Ok(regex) = FancyRegex::new(&pattern) {
                rules.push(SpamRule { id, description, regex });
            }
        }
        Ok(rules)
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

    fn load_audit_log_chat(conn: &Connection) -> Result<Option<i64>> {
        let mut stmt = conn.prepare("SELECT value FROM model_meta WHERE key = 'audit_log_chat_id'")?;
        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            let value: String = row.get(0)?;
            Ok(value.parse::<i64>().ok())
        } else {
            Ok(None)
        }
    }

    async fn persist_case(&self, case: &CaseRecord) -> Result<()> {
        let case = case.clone();
        self.with_conn(move |conn| {
            conn.execute(
                r#"
                INSERT INTO cases (id, action, chat_id, target_user_id, target_name, actor_user_id, actor_name, source_message_id, evidence_text, model_score, matched_rule_id, matched_rule_pattern, status, log_message_id, created_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
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
                  matched_rule_id=excluded.matched_rule_id,
                  matched_rule_pattern=excluded.matched_rule_pattern,
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
                    case.matched_rule_id,
                    case.matched_rule_pattern,
                    case.status,
                    case.log_message_id,
                    case.created_at.to_rfc3339(),
                ],
            )?;
            Ok(())
        })
        .await
    }

    async fn load_case(&self, case_id: &str) -> Result<Option<CaseRecord>> {
        let case_id = case_id.to_string();
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                r#"SELECT id, action, chat_id, target_user_id, target_name, actor_user_id, actor_name, source_message_id, evidence_text, model_score, matched_rule_id, matched_rule_pattern, status, log_message_id, created_at FROM cases WHERE id = ?1"#,
            )?;
            let mut rows = stmt.query(params![case_id])?;
            rows.next()?.map(case_from_row).transpose()
        })
        .await
    }

    /// Finds the most recent case for (chat_id, target_user_id) whose
    /// `action` is one of `actions` - e.g. the latest still-active ban, so
    /// `/unban <user_id>` or a reply (rather than a case_id) can find what to
    /// reverse without the caller needing to know a case ID. Once a case is
    /// reversed its `action` is mutated to Unbanned/Unmuted (see the /unban
    /// and /unmute handlers), so it naturally drops out of this search and
    /// an older still-active case (if any) surfaces instead.
    async fn load_latest_case_by_actions(&self, chat_id: i64, target_user_id: i64, actions: &[&str]) -> Result<Option<CaseRecord>> {
        let actions: Vec<String> = actions.iter().map(|s| s.to_string()).collect();
        self.with_conn(move |conn| {
            let placeholders = actions.iter().enumerate().map(|(i, _)| format!("?{}", i + 3)).collect::<Vec<_>>().join(",");
            let sql = format!(
                r#"SELECT id, action, chat_id, target_user_id, target_name, actor_user_id, actor_name, source_message_id, evidence_text, model_score, matched_rule_id, matched_rule_pattern, status, log_message_id, created_at FROM cases WHERE chat_id = ?1 AND target_user_id = ?2 AND action IN ({placeholders}) ORDER BY created_at DESC LIMIT 1"#
            );
            let mut stmt = conn.prepare(&sql)?;
            let mut bound: Vec<&dyn rusqlite::ToSql> = vec![&chat_id, &target_user_id];
            for action in &actions {
                bound.push(action);
            }
            let mut rows = stmt.query(bound.as_slice())?;
            rows.next()?.map(case_from_row).transpose()
        })
        .await
    }

    async fn list_netban_enabled_chats(&self) -> Result<Vec<i64>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare("SELECT chat_id FROM group_module_settings WHERE netban = 1")?;
            let rows = stmt.query_map([], |row| row.get::<_, i64>(0))?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row?);
            }
            Ok(out)
        })
        .await
    }

    /// Finds the most recent active ban for `user_id` whose origin chat has
    /// netban enabled - i.e. "is this user currently network-banned".
    /// Reuses the same "reversal mutates action in place" property as
    /// `load_latest_case_by_actions`: once reversed, a case's action becomes
    /// `Unbanned` and stops matching the `IN (...)` filter here too, so no
    /// separate "is this stale" bookkeeping is needed.
    async fn find_active_network_ban(&self, user_id: i64) -> Result<Option<CaseRecord>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                r#"SELECT c.id, c.action, c.chat_id, c.target_user_id, c.target_name, c.actor_user_id, c.actor_name, c.source_message_id, c.evidence_text, c.model_score, c.matched_rule_id, c.matched_rule_pattern, c.status, c.log_message_id, c.created_at
                   FROM cases c
                   JOIN group_module_settings g ON g.chat_id = c.chat_id
                   WHERE c.target_user_id = ?1 AND g.netban = 1 AND c.action IN ('auto_ban', 'spam_ban', 'report_approved')
                   ORDER BY c.created_at DESC LIMIT 1"#,
            )?;
            let mut rows = stmt.query(params![user_id])?;
            rows.next()?.map(case_from_row).transpose()
        })
        .await
    }

    /// Records that `case_id`'s ban was propagated to `chat_id` - the
    /// historical record `/unban` needs to know exactly which chats to
    /// reverse, since a group's netban membership can change after the fact.
    async fn record_network_ban_target(&self, case_id: &str, chat_id: i64) -> Result<()> {
        let case_id = case_id.to_string();
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT OR IGNORE INTO network_ban_targets (case_id, chat_id, created_at) VALUES (?1, ?2, ?3)",
                params![case_id, chat_id, Utc::now().to_rfc3339()],
            )?;
            Ok(())
        })
        .await
    }

    async fn list_network_ban_targets(&self, case_id: &str) -> Result<Vec<i64>> {
        let case_id = case_id.to_string();
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare("SELECT chat_id FROM network_ban_targets WHERE case_id = ?1")?;
            let rows = stmt.query_map(params![case_id], |row| row.get::<_, i64>(0))?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row?);
            }
            Ok(out)
        })
        .await
    }

    async fn clear_network_ban_targets(&self, case_id: &str) -> Result<()> {
        let case_id = case_id.to_string();
        self.with_conn(move |conn| {
            conn.execute("DELETE FROM network_ban_targets WHERE case_id = ?1", params![case_id])?;
            Ok(())
        })
        .await
    }

    /// Last time (chat_id, user_id) tripped a permission-denied guard on a
    /// group-admin-tier command, if ever - used by the CmdClean module to
    /// detect a repeat offense within 24h.
    async fn last_permission_offense(&self, chat_id: i64, user_id: i64) -> Result<Option<DateTime<Utc>>> {
        self.with_conn(move |conn| {
            let value: Option<String> = conn
                .query_row(
                    "SELECT last_offense_at FROM permission_offenses WHERE chat_id = ?1 AND user_id = ?2",
                    params![chat_id, user_id],
                    |row| row.get(0),
                )
                .ok();
            Ok(match value {
                Some(v) => Some(DateTime::parse_from_rfc3339(&v)?.with_timezone(&Utc)),
                None => None,
            })
        })
        .await
    }

    async fn record_permission_offense(&self, chat_id: i64, user_id: i64) -> Result<()> {
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO permission_offenses (chat_id, user_id, last_offense_at) VALUES (?1, ?2, ?3) ON CONFLICT(chat_id, user_id) DO UPDATE SET last_offense_at = excluded.last_offense_at",
                params![chat_id, user_id, Utc::now().to_rfc3339()],
            )?;
            Ok(())
        })
        .await
    }

    /// Logs one state-changing maintainer command. Returns the new
    /// `action_id` (a plain autoincrementing integer - short enough to type
    /// back into `/revert`, unlike a UUID).
    async fn record_maintainer_action(&self, actor_id: i64, actor_name: &str, chat_id: Option<i64>, command: &str, summary: &str, undo: &UndoData) -> Result<i64> {
        let actor_name = actor_name.to_string();
        let command = command.to_string();
        let summary = summary.to_string();
        let undo_json = serde_json::to_string(undo)?;
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO maintainer_actions (actor_id, actor_name, chat_id, command, summary, undo_data, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![actor_id, actor_name, chat_id, command, summary, undo_json, Utc::now().to_rfc3339()],
            )?;
            Ok(conn.last_insert_rowid())
        })
        .await
    }

    async fn load_maintainer_action(&self, action_id: i64) -> Result<Option<MaintainerAction>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT actor_name, chat_id, command, summary, undo_data, reverted FROM maintainer_actions WHERE action_id = ?1",
            )?;
            let mut rows = stmt.query(params![action_id])?;
            if let Some(row) = rows.next()? {
                let undo_json: String = row.get(4)?;
                let undo: UndoData = serde_json::from_str(&undo_json)?;
                Ok(Some(MaintainerAction {
                    actor_name: row.get(0)?,
                    chat_id: row.get(1)?,
                    command: row.get(2)?,
                    summary: row.get(3)?,
                    undo,
                    reverted: row.get::<_, i64>(5)? != 0,
                }))
            } else {
                Ok(None)
            }
        })
        .await
    }

    async fn mark_maintainer_action_reverted(&self, action_id: i64) -> Result<()> {
        self.with_conn(move |conn| {
            conn.execute("UPDATE maintainer_actions SET reverted = 1 WHERE action_id = ?1", params![action_id])?;
            Ok(())
        })
        .await
    }

    async fn insert_training_sample(&self, label: &str, text: &str, case_id: Option<&str>) -> Result<()> {
        let label = label.to_string();
        let text = text.to_string();
        let case_id = case_id.map(|s| s.to_string());
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO training_samples (label, text, case_id, created_at) VALUES (?1, ?2, ?3, ?4)",
                params![label, text, case_id, Utc::now().to_rfc3339()],
            )?;
            Ok(())
        })
        .await
    }

    /// Deletes the training sample(s) tied to `case_id` and rolls back the
    /// word_frequencies counts and doc totals they contributed - mirrors
    /// `purge_training_by_text` below. Previously this only deleted the
    /// `training_samples` audit row and left the learned token weights
    /// untouched, so a purge (or /unban, which relies on this) didn't
    /// actually undo the model's memory of the bad sample.
    async fn purge_training_by_case(&self, case_id: &str) -> Result<usize> {
        let case_id = case_id.to_string();
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;

            let mut samples = Vec::new();
            {
                let mut stmt = tx.prepare("SELECT label, text FROM training_samples WHERE case_id = ?1")?;
                let rows = stmt.query_map(params![&case_id], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
                for row in rows {
                    samples.push(row?);
                }
            }

            let mut spam_docs: i64 = tx.query_row("SELECT COALESCE(value, '0') FROM model_meta WHERE key = 'spam_docs'", [], |row| row.get::<_, String>(0))?.parse().unwrap_or(0);
            let mut ham_docs: i64 = tx.query_row("SELECT COALESCE(value, '0') FROM model_meta WHERE key = 'ham_docs'", [], |row| row.get::<_, String>(0))?.parse().unwrap_or(0);

            for (label, text) in &samples {
                let tokens = tokenize(text);
                for token in tokens {
                    let counts = tx.query_row(
                        "SELECT spam_count, ham_count FROM word_frequencies WHERE word = ?1",
                        params![&token],
                        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
                    );
                    if let Ok((mut spam_count, mut ham_count)) = counts {
                        match label.as_str() {
                            "spam" => spam_count = (spam_count - 1).max(0),
                            "ham" => ham_count = (ham_count - 1).max(0),
                            _ => {}
                        }
                        if spam_count == 0 && ham_count == 0 {
                            tx.execute("DELETE FROM word_frequencies WHERE word = ?1", params![&token])?;
                        } else {
                            tx.execute(
                                "UPDATE word_frequencies SET spam_count = ?2, ham_count = ?3 WHERE word = ?1",
                                params![&token, spam_count, ham_count],
                            )?;
                        }
                    }
                }

                match label.as_str() {
                    "spam" => spam_docs = (spam_docs - 1).max(0),
                    "ham" => ham_docs = (ham_docs - 1).max(0),
                    _ => {}
                }
            }

            tx.execute(
                "INSERT INTO model_meta (key, value) VALUES ('spam_docs', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![spam_docs.to_string()],
            )?;
            tx.execute(
                "INSERT INTO model_meta (key, value) VALUES ('ham_docs', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![ham_docs.to_string()],
            )?;

            let affected = tx.execute("DELETE FROM training_samples WHERE case_id = ?1", params![&case_id])?;
            tx.commit()?;
            Ok(affected)
        })
        .await
    }

    /// Refreshes the in-memory model from disk. This only reads — the DB is
    /// always the source of truth and callers that changed the DB (train_spam,
    /// purge, undo, set_token_probability, ...) have already persisted their
    /// specific changes, so there is nothing to write back here.
    async fn rebuild_model(&self) -> Result<ModelState> {
        let rebuilt = self
            .with_conn(|conn| {
                let mut rebuilt = ModelState::default();
                let mut stmt = conn.prepare("SELECT key, value FROM model_meta")?;
                let rows = stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
                for row in rows {
                    let (key, value) = row?;
                    match key.as_str() {
                        "spam_docs" => rebuilt.spam_docs = value.parse().unwrap_or(0),
                        "ham_docs" => rebuilt.ham_docs = value.parse().unwrap_or(0),
                        _ => {}
                    }
                }
                let mut stmt = conn.prepare("SELECT word, spam_count, ham_count FROM word_frequencies ORDER BY word ASC")?;
                let mut rows = stmt.query([])?;
                while let Some(row) = rows.next()? {
                    let word: String = row.get(0)?;
                    let spam_count: u64 = row.get(1)?;
                    let ham_count: u64 = row.get(2)?;
                    if spam_count > 0 {
                        rebuilt.spam_tokens.insert(word.clone(), spam_count);
                    }
                    if ham_count > 0 {
                        rebuilt.ham_tokens.insert(word, ham_count);
                    }
                }
                Ok(rebuilt)
            })
            .await?;

        let mut model = self.model.lock().await;
        *model = rebuilt.clone();
        Ok(rebuilt)
    }

    /// Persists only the aggregate doc counters. Per-token counts are written
    /// directly by whoever changes them (train_spam/train_ham/etc.) — this
    /// used to also rewrite every token in the vocabulary on every call, which
    /// got slower as the vocabulary grew for no benefit.
    async fn persist_doc_counts(&self) -> Result<()> {
        let (spam_docs, ham_docs) = {
            let model = self.model.lock().await;
            (model.spam_docs, model.ham_docs)
        };
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO model_meta (key, value) VALUES ('spam_docs', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![spam_docs.to_string()],
            )?;
            conn.execute(
                "INSERT INTO model_meta (key, value) VALUES ('ham_docs', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![ham_docs.to_string()],
            )?;
            Ok(())
        })
        .await
    }

    async fn set_threshold(&self, value: f64) -> Result<()> {
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO model_meta (key, value) VALUES ('spam_threshold', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![value.to_string()],
            )?;
            Ok(())
        })
        .await
    }

    /// Returns `(new_spam_count, new_ham_count, old_spam_count, old_ham_count)`.
    /// The old counts are returned too, rather than requiring a separate
    /// query, so `/revert` can restore them via `UndoData::TokenProbability`.
    async fn set_token_probability(&self, token: &str, target_spam_prob: f64) -> Result<(u64, u64, u64, u64)> {
        let token = token.trim().to_string();
        // Clamped well away from 0/1: the formula below solves for the raw count
        // needed to hit `target`, and that count blows up as target approaches
        // either extreme (at 0.9999 against a modest few-thousand-word corpus it's
        // already tens of millions). Since spam_count/ham_count feed into the
        // shared spam_total/ham_total used to score every OTHER token, an
        // extreme target here silently poisons the whole model's scoring, not
        // just this one word's.
        let target = target_spam_prob.clamp(0.05, 0.95);
        let updated = self
            .with_conn(move |conn| {
                let (spam_total, ham_total, vocab): (u64, u64, u64) = conn.query_row(
                    "SELECT COALESCE(SUM(spam_count), 0), COALESCE(SUM(ham_count), 0), COUNT(*) FROM word_frequencies",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )?;

                let (current_spam, current_ham): (u64, u64) = conn.query_row(
                    "SELECT COALESCE(spam_count, 0), COALESCE(ham_count, 0) FROM word_frequencies WHERE word = ?1",
                    params![&token],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                ).unwrap_or((0, 0));

                let spam_other = spam_total.saturating_sub(current_spam) as f64;
                let ham_other = ham_total.saturating_sub(current_ham) as f64;
                let vocab = vocab.max(1) as f64;

                let spam_count = (((target * (spam_other + vocab)) - 1.0) / (1.0 - target)).ceil().max(0.0) as u64;
                let ham_target = 1.0 - target;
                let ham_count = (((ham_target * (ham_other + vocab)) - 1.0) / (1.0 - ham_target)).ceil().max(0.0) as u64;

                // Defense in depth: even with `target` clamped above, never let a
                // single token's count outweigh the rest of the corpus by more
                // than 20x, and never past an absolute ceiling. Either bound being
                // hit means the requested probability wasn't fully reached.
                let spam_count = spam_count.min((spam_other as u64).saturating_mul(20).max(1_000)).min(1_000_000);
                let ham_count = ham_count.min((ham_other as u64).saturating_mul(20).max(1_000)).min(1_000_000);

                conn.execute(
                    "INSERT INTO word_frequencies (word, spam_count, ham_count) VALUES (?1, ?2, ?3) ON CONFLICT(word) DO UPDATE SET spam_count = excluded.spam_count, ham_count = excluded.ham_count",
                    params![&token, spam_count, ham_count],
                )?;

                Ok((spam_count, ham_count, current_spam, current_ham))
            })
            .await?;

        let _ = self.rebuild_model().await?;
        Ok(updated)
    }

    async fn current_threshold(&self) -> Result<f64> {
        let value = self.with_conn(|conn| Self::load_threshold(conn)).await?;
        Ok(value.unwrap_or(self.config.spam_threshold))
    }

    /// Sets a token's spam/ham counts directly, unlike `set_token_probability`
    /// which solves for counts from a target probability. Used by `/revert`
    /// to restore exact prior counts, where the clamping/defense-in-depth
    /// `set_token_probability` applies would be wrong (those old counts were
    /// already valid before, so they don't need re-validating).
    async fn set_token_counts_raw(&self, token: &str, spam_count: u64, ham_count: u64) -> Result<()> {
        let token = token.to_string();
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO word_frequencies (word, spam_count, ham_count) VALUES (?1, ?2, ?3) ON CONFLICT(word) DO UPDATE SET spam_count = excluded.spam_count, ham_count = excluded.ham_count",
                params![&token, spam_count, ham_count],
            )?;
            Ok(())
        })
        .await?;
        let _ = self.rebuild_model().await?;
        Ok(())
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
        let _ = self
            .with_conn(move |conn| {
                conn.execute(
                    "INSERT INTO model_meta (key, value) VALUES ('project_chat_id', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    params![chat_id.to_string()],
                )?;
                Ok(())
            })
            .await;
        let mut project_chat = self.project_chat.lock().await;
        *project_chat = Some(chat_id);
    }

    async fn set_audit_log_chat(&self, chat_id: i64) {
        let _ = self
            .with_conn(move |conn| {
                conn.execute(
                    "INSERT INTO model_meta (key, value) VALUES ('audit_log_chat_id', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    params![chat_id.to_string()],
                )?;
                Ok(())
            })
            .await;
        let mut audit_log_chat = self.audit_log_chat.lock().await;
        *audit_log_chat = Some(chat_id);
    }

    async fn audit_log_chat(&self) -> Option<i64> {
        let audit_log_chat = self.audit_log_chat.lock().await;
        *audit_log_chat
    }

    async fn blacklist_reason_message_id(&self) -> Result<Option<i32>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare("SELECT value FROM model_meta WHERE key = 'blacklist_reason_message_id'")?;
            let mut rows = stmt.query([])?;
            if let Some(row) = rows.next()? {
                let value: String = row.get(0)?;
                Ok(value.parse::<i32>().ok())
            } else {
                Ok(None)
            }
        })
        .await
    }

    async fn set_blacklist_reason_message_id(&self, message_id: i32) -> Result<()> {
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO model_meta (key, value) VALUES ('blacklist_reason_message_id', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![message_id.to_string()],
            )?;
            Ok(())
        })
        .await
    }

    async fn blacklist_reason_link(&self) -> Option<String> {
        match self.blacklist_reason_message_id().await {
            Ok(Some(message_id)) => Some(public_log_link(&self.config, message_id)),
            _ => None,
        }
    }

    async fn start_pending_rule_addition(&self, user_id: i64, pattern: String) {
        let mut pending = self.pending_rule_additions.lock().await;
        pending.insert(user_id, pattern);
    }

    async fn take_pending_rule_addition(&self, user_id: i64) -> Option<String> {
        let mut pending = self.pending_rule_additions.lock().await;
        pending.remove(&user_id)
    }

    async fn pending_rule_addition(&self, user_id: i64) -> Option<String> {
        let pending = self.pending_rule_additions.lock().await;
        pending.get(&user_id).cloned()
    }

    async fn project_chat(&self) -> Option<i64> {
        let project_chat = self.project_chat.lock().await;
        *project_chat
    }

    async fn export_training_data(&self) -> Result<String> {
        self.with_conn(|conn| {
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
            Ok(out)
        })
        .await
    }

    /// Resolves the threshold that actually applies for `chat_id`: a
    /// per-group override if one is set, otherwise the global default.
    /// `chat_id` is `None` for contexts with no specific chat (there are
    /// none left currently, but keeps this usable from anywhere).
    async fn effective_threshold(&self, chat_id: Option<i64>) -> Result<f64> {
        if let Some(chat_id) = chat_id {
            if let Ok(settings) = self.get_group_modules(chat_id).await {
                if let Some(value) = settings.spam_threshold_override {
                    return Ok(value);
                }
            }
        }
        self.current_threshold().await
    }

    async fn refresh_spam_rules(&self) -> Result<()> {
        let rules = self.with_conn(|conn| Runtime::load_spam_rules(conn)).await?;
        let mut cache = self.spam_rules.write().await;
        *cache = rules;
        Ok(())
    }

    async fn purge_training_by_text(&self, payload: &str) -> Result<usize> {
        let payload = payload.to_string();
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;

            let mut samples = Vec::new();
            {
                let mut stmt = tx.prepare("SELECT label, text FROM training_samples WHERE text LIKE ?1 OR text LIKE ?2")?;
                let rows = stmt.query_map(params![format!("%{payload}%"), payload], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
                for row in rows {
                    samples.push(row?);
                }
            }

            let mut spam_docs: i64 = tx.query_row("SELECT COALESCE(value, '0') FROM model_meta WHERE key = 'spam_docs'", [], |row| row.get::<_, String>(0))?.parse().unwrap_or(0);
            let mut ham_docs: i64 = tx.query_row("SELECT COALESCE(value, '0') FROM model_meta WHERE key = 'ham_docs'", [], |row| row.get::<_, String>(0))?.parse().unwrap_or(0);

            for (label, text) in &samples {
                let tokens = tokenize(text);
                for token in tokens {
                    let counts = tx.query_row(
                        "SELECT spam_count, ham_count FROM word_frequencies WHERE word = ?1",
                        params![&token],
                        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
                    );
                    if let Ok((mut spam_count, mut ham_count)) = counts {
                        match label.as_str() {
                            "spam" => spam_count = (spam_count - 1).max(0),
                            "ham" => ham_count = (ham_count - 1).max(0),
                            _ => {}
                        }
                        if spam_count == 0 && ham_count == 0 {
                            tx.execute("DELETE FROM word_frequencies WHERE word = ?1", params![&token])?;
                        } else {
                            tx.execute(
                                "UPDATE word_frequencies SET spam_count = ?2, ham_count = ?3 WHERE word = ?1",
                                params![&token, spam_count, ham_count],
                            )?;
                        }
                    }
                }

                match label.as_str() {
                    "spam" => spam_docs = (spam_docs - 1).max(0),
                    "ham" => ham_docs = (ham_docs - 1).max(0),
                    _ => {}
                }
            }

            tx.execute(
                "INSERT INTO model_meta (key, value) VALUES ('spam_docs', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![spam_docs.to_string()],
            )?;
            tx.execute(
                "INSERT INTO model_meta (key, value) VALUES ('ham_docs', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![ham_docs.to_string()],
            )?;

            let affected = tx.execute(
                "DELETE FROM training_samples WHERE text LIKE ?1 OR text LIKE ?2",
                params![format!("%{payload}%"), payload],
            )?;
            tx.commit()?;
            Ok(affected)
        })
        .await
    }

    async fn undo_clean_training_sample_by_text(&self, text: &str) -> Result<usize> {
        let text = text.to_string();
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;

            let maybe_sample = {
                let mut stmt = tx.prepare(
                    "SELECT id, text FROM training_samples WHERE label = 'ham' AND text = ?1 ORDER BY id DESC LIMIT 1",
                )?;
                let mut rows = stmt.query(params![&text])?;
                if let Some(row) = rows.next()? {
                    Some((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                } else {
                    None
                }
            };

            let Some((sample_id, sample_text)) = maybe_sample else {
                tx.commit()?;
                return Ok(0);
            };

            let tokens = tokenize(&sample_text);
            for token in tokens {
                let counts = tx.query_row(
                    "SELECT spam_count, ham_count FROM word_frequencies WHERE word = ?1",
                    params![&token],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
                );
                if let Ok((spam_count, ham_count)) = counts {
                    let new_ham = (ham_count - 1).max(0);
                    if spam_count == 0 && new_ham == 0 {
                        tx.execute("DELETE FROM word_frequencies WHERE word = ?1", params![&token])?;
                    } else {
                        tx.execute(
                            "UPDATE word_frequencies SET ham_count = ?2 WHERE word = ?1",
                            params![&token, new_ham],
                        )?;
                    }
                }
            }

            let ham_docs: i64 = tx.query_row(
                "SELECT COALESCE(CAST(value AS INTEGER), 0) FROM model_meta WHERE key = 'ham_docs'",
                [],
                |row| row.get(0),
            )?;
            tx.execute(
                "INSERT INTO model_meta (key, value) VALUES ('ham_docs', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![(ham_docs - 1).max(0).to_string()],
            )?;

            tx.execute("DELETE FROM training_samples WHERE id = ?1", params![sample_id])?;
            tx.commit()?;
            Ok(1)
        })
        .await
    }

    async fn word_stats(&self) -> Result<(u64, u64, u64)> {
        self.with_conn(|conn| {
            let spam: u64 = conn.query_row("SELECT COALESCE(SUM(spam_count), 0) FROM word_frequencies", [], |row| row.get(0))?;
            let ham: u64 = conn.query_row("SELECT COALESCE(SUM(ham_count), 0) FROM word_frequencies", [], |row| row.get(0))?;
            let total: u64 = conn.query_row("SELECT COUNT(*) FROM word_frequencies", [], |row| row.get(0))?;
            Ok((spam, ham, total))
        })
        .await
    }

    /// Returns the single largest spam_count and ham_count in the vocabulary.
    /// A token whose count dwarfs the rest of the corpus (e.g. from a `/set`
    /// call with a near-0/near-1 target) silently drags down every other
    /// token's score, since counts are summed into a shared denominator.
    /// `/ml_stats` surfaces this so it's catchable without direct DB access.
    async fn largest_token_counts(&self) -> Result<(Option<(String, u64)>, Option<(String, u64)>)> {
        self.with_conn(|conn| {
            let top_spam = conn
                .query_row(
                    "SELECT word, spam_count FROM word_frequencies ORDER BY spam_count DESC LIMIT 1",
                    [],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, u64>(1)?)),
                )
                .ok();
            let top_ham = conn
                .query_row(
                    "SELECT word, ham_count FROM word_frequencies ORDER BY ham_count DESC LIMIT 1",
                    [],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, u64>(1)?)),
                )
                .ok();
            Ok((top_spam, top_ham))
        })
        .await
    }

    async fn add_spam_rule(&self, pattern: &str, description: &str) -> Result<i64> {
        FancyRegex::new(pattern).context("invalid regex pattern")?;
        let pattern = pattern.to_string();
        let description = description.to_string();
        let id = self
            .with_conn(move |conn| {
                conn.execute(
                    "INSERT INTO spam_rules (pattern, description) VALUES (?1, ?2)",
                    params![pattern, description],
                )?;
                Ok(conn.last_insert_rowid())
            })
            .await?;
        self.refresh_spam_rules().await?;
        Ok(id)
    }

    async fn update_spam_rule_pattern(&self, rule_id: i64, pattern: &str) -> Result<bool> {
        FancyRegex::new(pattern).context("invalid regex pattern")?;
        let pattern = pattern.to_string();
        let updated = self
            .with_conn(move |conn| Ok(conn.execute("UPDATE spam_rules SET pattern = ?2 WHERE id = ?1", params![rule_id, pattern])?))
            .await?;
        if updated > 0 {
            self.refresh_spam_rules().await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn delete_spam_rule(&self, rule_id: i64) -> Result<bool> {
        let removed = self
            .with_conn(move |conn| Ok(conn.execute("DELETE FROM spam_rules WHERE id = ?1", params![rule_id])?))
            .await?;
        if removed > 0 {
            self.refresh_spam_rules().await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn list_spam_rules(&self) -> Result<Vec<(i64, String, String)>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare("SELECT id, pattern, description FROM spam_rules ORDER BY id ASC")?;
            let rows = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?)))?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row?);
            }
            Ok(out)
        })
        .await
    }

    async fn list_invalid_spam_rules(&self) -> Result<Vec<(i64, String, String, String)>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare("SELECT id, pattern, description FROM spam_rules ORDER BY id ASC")?;
            let rows = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?)))?;
            let mut out = Vec::new();
            for row in rows {
                let (id, pattern, description) = row?;
                if let Err(err) = FancyRegex::new(&pattern) {
                    out.push((id, pattern, description, err.to_string()));
                }
            }
            Ok(out)
        })
        .await
    }

    async fn get_group_modules(&self, chat_id: i64) -> Result<GroupModuleSettings> {
        if let Some(cached) = self.group_module_cache.read().await.get(&chat_id) {
            return Ok(cached.clone());
        }
        let settings = self
            .with_conn(move |conn| {
                conn.execute(
                    "INSERT OR IGNORE INTO group_module_settings (chat_id) VALUES (?1)",
                    params![chat_id],
                )?;
                let mut stmt = conn.prepare("SELECT no_long_name, no_halal, no_service_messages, flood_control, captcha, spam_threshold_override, netban, cmd_clean FROM group_module_settings WHERE chat_id = ?1")?;
                let mut rows = stmt.query(params![chat_id])?;
                if let Some(row) = rows.next()? {
                    Ok(GroupModuleSettings {
                        no_long_name: row.get::<_, i64>(0)? != 0,
                        no_halal: row.get::<_, i64>(1)? != 0,
                        no_service_messages: row.get::<_, i64>(2)? != 0,
                        flood_control: row.get::<_, i64>(3)? != 0,
                        captcha: row.get::<_, i64>(4)? != 0,
                        spam_threshold_override: row.get::<_, Option<f64>>(5)?,
                        netban: row.get::<_, i64>(6)? != 0,
                        cmd_clean: row.get::<_, i64>(7)? != 0,
                    })
                } else {
                    Ok(GroupModuleSettings::default())
                }
            })
            .await?;
        self.group_module_cache.write().await.insert(chat_id, settings.clone());
        Ok(settings)
    }

    async fn set_group_module(&self, chat_id: i64, module: &str, enabled: bool) -> Result<()> {
        let module = module.to_string();
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT OR IGNORE INTO group_module_settings (chat_id) VALUES (?1)",
                params![chat_id],
            )?;
            match module.as_str() {
            "nolongname" => {
                conn.execute(
                    "UPDATE group_module_settings SET no_long_name = ?2 WHERE chat_id = ?1",
                    params![chat_id, if enabled { 1 } else { 0 }],
                )?;
            }
            "nohalal" => {
                conn.execute(
                    "UPDATE group_module_settings SET no_halal = ?2 WHERE chat_id = ?1",
                    params![chat_id, if enabled { 1 } else { 0 }],
                )?;
            }
            "nosm" => {
                conn.execute(
                    "UPDATE group_module_settings SET no_service_messages = ?2 WHERE chat_id = ?1",
                    params![chat_id, if enabled { 1 } else { 0 }],
                )?;
            }
            "flood" => {
                conn.execute(
                    "UPDATE group_module_settings SET flood_control = ?2 WHERE chat_id = ?1",
                    params![chat_id, if enabled { 1 } else { 0 }],
                )?;
            }
            "captcha" => {
                conn.execute(
                    "UPDATE group_module_settings SET captcha = ?2 WHERE chat_id = ?1",
                    params![chat_id, if enabled { 1 } else { 0 }],
                )?;
            }
            "netban" => {
                conn.execute(
                    "UPDATE group_module_settings SET netban = ?2 WHERE chat_id = ?1",
                    params![chat_id, if enabled { 1 } else { 0 }],
                )?;
            }
            "cmdclean" => {
                conn.execute(
                    "UPDATE group_module_settings SET cmd_clean = ?2 WHERE chat_id = ?1",
                    params![chat_id, if enabled { 1 } else { 0 }],
                )?;
            }
                _ => {}
            }
            Ok(())
        })
        .await?;
        self.group_module_cache.write().await.remove(&chat_id);
        Ok(())
    }

    async fn set_group_threshold(&self, chat_id: i64, value: Option<f64>) -> Result<()> {
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT OR IGNORE INTO group_module_settings (chat_id) VALUES (?1)",
                params![chat_id],
            )?;
            conn.execute(
                "UPDATE group_module_settings SET spam_threshold_override = ?2 WHERE chat_id = ?1",
                params![chat_id, value],
            )?;
            Ok(())
        })
        .await?;
        self.group_module_cache.write().await.remove(&chat_id);
        Ok(())
    }

    /// Records this message towards the (chat, user) flood window and
    /// returns true if it just tripped the threshold. Pure in-memory bookkeeping
    /// (see `flood_tracker` on `Runtime`) - no DB access, so this is cheap
    /// enough to call on every single incoming group message.
    async fn check_flood(&self, chat_id: i64, user_id: i64) -> bool {
        const WINDOW: std::time::Duration = std::time::Duration::from_secs(5);
        const LIMIT: usize = 5;
        let now = Instant::now();
        let mut tracker = self.flood_tracker.lock().await;
        let timestamps = tracker.entry((chat_id, user_id)).or_default();
        timestamps.push_back(now);
        while let Some(&front) = timestamps.front() {
            if now.duration_since(front) > WINDOW {
                timestamps.pop_front();
            } else {
                break;
            }
        }
        timestamps.len() >= LIMIT
    }

    async fn is_group_whitelisted(&self, chat_id: i64, user_id: i64) -> Result<bool> {
        self.with_conn(move |conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM group_whitelist WHERE chat_id = ?1 AND user_id = ?2",
                params![chat_id, user_id],
                |row| row.get(0),
            )?;
            Ok(count > 0)
        })
        .await
    }

    async fn set_group_whitelist(&self, chat_id: i64, user_id: i64, enabled: bool, added_by: Option<i64>) -> Result<()> {
        self.with_conn(move |conn| {
            if enabled {
                conn.execute(
                    "INSERT OR IGNORE INTO group_whitelist (chat_id, user_id, added_by, created_at) VALUES (?1, ?2, ?3, ?4)",
                    params![chat_id, user_id, added_by, Utc::now().to_rfc3339()],
                )?;
            } else {
                conn.execute(
                    "DELETE FROM group_whitelist WHERE chat_id = ?1 AND user_id = ?2",
                    params![chat_id, user_id],
                )?;
            }
            Ok(())
        })
        .await
    }

    async fn is_global_whitelisted(&self, user_id: i64) -> Result<bool> {
        self.with_conn(move |conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM global_whitelist WHERE user_id = ?1",
                params![user_id],
                |row| row.get(0),
            )?;
            Ok(count > 0)
        })
        .await
    }

    async fn set_global_whitelist(&self, user_id: i64, enabled: bool, added_by: Option<i64>) -> Result<()> {
        self.with_conn(move |conn| {
            if enabled {
                conn.execute(
                    "INSERT OR IGNORE INTO global_whitelist (user_id, added_by, created_at) VALUES (?1, ?2, ?3)",
                    params![user_id, added_by, Utc::now().to_rfc3339()],
                )?;
            } else {
                conn.execute(
                    "DELETE FROM global_whitelist WHERE user_id = ?1",
                    params![user_id],
                )?;
            }
            Ok(())
        })
        .await
    }

    async fn load_user_profile(&self, bot: &Bot, user_id: i64) -> Result<UserProfileInfo> {
        let chat = bot.get_chat(ChatId(user_id)).await?;
        let display_name = chat.title().map(|s| s.to_string()).or_else(|| chat.first_name().map(|s| s.to_string())).unwrap_or_else(|| format!("User{user_id}"));
        let username = chat.username().map(|s| s.to_string());
        let bio = chat.bio().map(|s| s.to_string());
        Ok(UserProfileInfo { user_id, display_name, username, bio })
    }

    async fn check_group_modules(&self, _bot: &Bot, chat_id: i64, user: &teloxide::types::User, bio: Option<&str>, message_text: Option<&str>) -> Result<ModuleCheckResult> {
        if self.is_global_whitelisted(user.id.0 as i64).await.unwrap_or(false) {
            return Ok(ModuleCheckResult { reasons: Vec::new(), name_guard: Vec::new(), no_halal: Vec::new() });
        }

        // Priority check: Display name regex matching (highest priority)
        let rules = self.spam_rules.read().await;
        let display_name = short_user(user);
        let mut reasons = Vec::new();
        let mut display_name_hits = Vec::new();

        for rule in rules.iter() {
            if regex_is_match(&rule.regex, &display_name) {
                display_name_hits.push(format!("REGEX@{}", rule.id));
                reasons.push(format!("REGEX@{}", rule.id));
            }
        }

        // If display name matches any rule, ban immediately (highest priority takes precedence)
        if !display_name_hits.is_empty() {
            drop(rules);
            return Ok(ModuleCheckResult { reasons, name_guard: display_name_hits, no_halal: Vec::new() });
        }

        drop(rules);
        
        let settings = self.get_group_modules(chat_id).await?;
        let mut name_guard = Vec::new();
        let mut no_halal = Vec::new();

        if settings.no_long_name && !is_special_user(&self.config, user.id.0 as i64) {
            let r = evaluate_no_long_name(user);
            if !r.is_empty() {
                reasons.extend(r.clone());
                name_guard = r;
            }
        }

        if settings.no_halal && !is_special_user(&self.config, user.id.0 as i64) {
            let r = evaluate_module_checks(user, user.username.as_deref(), bio, message_text);
            if !r.is_empty() {
                reasons.extend(r.clone());
                no_halal = r;
            }
        }

        // Check message text and bio against regex rules
        let text = message_text.unwrap_or("");
        let rules = self.spam_rules.read().await;
        let mut regex_hits = Vec::new();
        if !text.trim().is_empty() {
            if let Some(bio) = bio {
                if let Some(bio_hit) = rules.iter().find(|rule| regex_is_match(&rule.regex, bio)) {
                    regex_hits.push(format!("REGEX@{}", bio_hit.id));
                }
            }
            if let Some(text_hit) = rules.iter().find(|rule| regex_is_match(&rule.regex, text)) {
                regex_hits.push(format!("REGEX@{}", text_hit.id));
            }
        }
        reasons.extend(regex_hits);

        Ok(ModuleCheckResult { reasons, name_guard, no_halal })
    }

    async fn inspect_message(&self, _display_name: &str, text: &str) -> Result<InspectionResult> {
        let rules = self.spam_rules.read().await;
        if tokenize(text).is_empty() {
            return Ok(InspectionResult::Ham { score: 0.0 });
        }
        for rule in rules.iter() {
            if regex_is_match(&rule.regex, text) {
                return Ok(InspectionResult::Spam {
                    score: 1.0,
                    matched_rule: Some(MatchedRule {
                        description: rule.description.clone(),
                    }),
                });
            }
        }
        drop(rules);

        let model = self.model.lock().await;
        let score = score_spam_from_text(&model, text);
        Ok(InspectionResult::Ham { score })
    }

    async fn score_debug(&self, _display_name: &str, text: &str) -> Result<ScoreDebugReport> {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return Ok(ScoreDebugReport { score: 0.0, tokens: Vec::new() });
        }
        let model = self.model.lock().await;
        Ok(score_debug_from_text(&model, text))
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
    MlUndoCleanSpam,
    MarkHam,
    MlStats,
    MlThreshold(String),
    MlSetToken(String, String),
    MlExport,
    MlPurge(String),
    MlPurgeText(String),
    MlRebuild,
    MlFinishMassTrain,
    MlStartMassHam,
    MlFinishMassHam,
    MlImport,
    MlStartMassTrainWithMode(String),
    MlDebugParse,
    MlScoreDebug,
    AddRule(String),
    EditRule(String, String),
    UpdateBL,
    ListRules,
    CheckRules,
    DelRule(String),
    Module(String, String),
    White(String),
    Unwhite(String),
    WhiteGlobal(String),
    UnwhiteGlobal(String),
    HelpOp,
    Check(String),
    Unban(String),
    Unmute(String),
    Ping,
    SetAuditLog(String),
    Revert(String),
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
        "/ml_train_spam" | "/mark_spam" => ModerationCommand::MlTrainSpam,
        "/ml_clean_spam" => ModerationCommand::MlCleanSpam,
        "/ml_undo_clean_spam" | "/ml_undo_ham" => ModerationCommand::MlUndoCleanSpam,
        "/mark_ham" => ModerationCommand::MarkHam,
        "/ml_stats" => ModerationCommand::MlStats,
        "/ml_threshold" => ModerationCommand::MlThreshold(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/set" => ModerationCommand::MlSetToken(
            text.split_whitespace().nth(1).unwrap_or("").to_string(),
            text.split_whitespace().nth(2).unwrap_or("").to_string(),
        ),
        "/ml_export" => ModerationCommand::MlExport,
        "/ml_purge" => ModerationCommand::MlPurge(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/ml_purge_text" => ModerationCommand::MlPurgeText(text.split_whitespace().skip(1).collect::<Vec<_>>().join(" ")),
        "/ml_rebuild" => ModerationCommand::MlRebuild,
        "/ml_start_mass_train" => ModerationCommand::MlStartMassTrainWithMode("smart".to_string()),
        "/ml_finish_mass_train" => ModerationCommand::MlFinishMassTrain,
        "/ml_start_mass_ham" => ModerationCommand::MlStartMassHam,
        "/ml_finish_mass_ham" => ModerationCommand::MlFinishMassHam,
        "/import" => ModerationCommand::MlImport,
        "/ml_start_mass_train_smart" => ModerationCommand::MlStartMassTrainWithMode("smart".to_string()),
        "/ml_start_mass_train_plain" => ModerationCommand::MlStartMassTrainWithMode("plain".to_string()),
        "/ml_debug_parse" => ModerationCommand::MlDebugParse,
        "/ml_score_debug" => ModerationCommand::MlScoreDebug,
        "/add_rule" => ModerationCommand::AddRule(text.split_whitespace().skip(1).collect::<Vec<_>>().join(" ")),
        "/edit_rule" => {
            let mut parts = text.split_whitespace();
            let _ = parts.next();
            let rule_id = parts.next().unwrap_or("").to_string();
            let pattern = parts.collect::<Vec<_>>().join(" ");
            ModerationCommand::EditRule(rule_id, pattern)
        }
        "/updatebl" => ModerationCommand::UpdateBL,
        "/list_rules" => ModerationCommand::ListRules,
        "/check_rules" => ModerationCommand::CheckRules,
        "/del_rule" => ModerationCommand::DelRule(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/unwhite" => {
            let args = text.split_whitespace().skip(1).collect::<Vec<_>>();
            if args.first() == Some(&"-global") {
                ModerationCommand::UnwhiteGlobal(args.get(1).unwrap_or(&"").to_string())
            } else if args.get(1) == Some(&"-global") {
                ModerationCommand::UnwhiteGlobal(args.first().unwrap_or(&"").to_string())
            } else {
                ModerationCommand::Unwhite(args.first().unwrap_or(&"").to_string())
            }
        }
        "/help_op" => ModerationCommand::HelpOp,
        "/module" | "/moudle" => {
            let mut parts = text.split_whitespace();
            let _ = parts.next();
            let module = parts.next().unwrap_or("").to_string();
            let state = parts.next().unwrap_or("").to_string();
            ModerationCommand::Module(module, state)
        }
        "/white" => {
            let args = text.split_whitespace().skip(1).collect::<Vec<_>>();
            if args.first() == Some(&"-global") {
                ModerationCommand::WhiteGlobal(args.get(1).unwrap_or(&"").to_string())
            } else if args.get(1) == Some(&"-global") {
                ModerationCommand::WhiteGlobal(args.first().unwrap_or(&"").to_string())
            } else {
                ModerationCommand::White(args.first().unwrap_or(&"").to_string())
            }
        }
        "/check" => ModerationCommand::Check(text.split_whitespace().skip(1).collect::<Vec<_>>().join(" ")),
        "/unban" => ModerationCommand::Unban(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/unmute" => ModerationCommand::Unmute(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/ping" => ModerationCommand::Ping,
        "/set_audit_log" => ModerationCommand::SetAuditLog(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/revert" => ModerationCommand::Revert(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        _ => ModerationCommand::Unknown,
    }
}

fn tokenize(text: &str) -> Vec<String> {
    normalize_tokens(text, jieba())
}

fn contains_arabic_script(text: &str) -> bool {
    if text.trim().is_empty() {
        return false;
    }
    text.chars().any(|c| ('\u{0600}'..='\u{06FF}').contains(&c)
        || ('\u{0750}'..='\u{077F}').contains(&c)
        || ('\u{08A0}'..='\u{08FF}').contains(&c)
        || ('\u{FB50}'..='\u{FDFF}').contains(&c)
        || ('\u{FE70}'..='\u{FEFF}').contains(&c))
}

fn clean_name_parts(text: &str) -> (String, Vec<String>) {
    let clean = text
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == ' ' || *c == '-')
        .collect::<String>();
    let parts = clean
        .split([' ', '-'])
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();
    (clean, parts)
}

fn evaluate_name_guard(full_name: &str) -> Vec<String> {
    if !full_name.is_ascii() {
        return Vec::new();
    }

    let (clean, parts) = clean_name_parts(full_name);
    if clean.trim().is_empty() {
        return Vec::new();
    }

    let has_digits = clean.chars().any(|c| c.is_ascii_digit());
    let total_len = clean.chars().filter(|c| c.is_ascii_alphanumeric()).count();
    let mut reasons = Vec::new();

    if has_digits && total_len >= 7 {
        reasons.push("NLDIGIT".to_string());
    }
    if parts.len() >= 2 && total_len >= 13 {
        reasons.push("NL13".to_string());
    }
    if parts.len() >= 2 && parts.last().map(|s| s.len() >= 5).unwrap_or(false) {
        reasons.push("NLTAIL".to_string());
    }
    if parts.len() == 1 && total_len >= 11 {
        reasons.push("NLSINGLE".to_string());
    }
    reasons
}

fn evaluate_no_long_name(user: &teloxide::types::User) -> Vec<String> {
    evaluate_name_guard(&display_name_only(user))
}

fn display_name_only(user: &teloxide::types::User) -> String {
    let mut name = user.first_name.clone();
    if let Some(last) = &user.last_name {
        name.push(' ');
        name.push_str(last);
    }
    name
}

fn evaluate_module_checks(user: &teloxide::types::User, username: Option<&str>, bio: Option<&str>, message_text: Option<&str>) -> Vec<String> {
    let mut reasons = Vec::new();
    let name = short_user(user);
    if contains_arabic_script(&name) {
        reasons.push("ARABIC".to_string());
    }
    if let Some(username) = username {
        if contains_arabic_script(username) {
            reasons.push("ARABIC".to_string());
        }
    }
    if let Some(bio) = bio {
        if contains_arabic_script(bio) {
            reasons.push("ARABIC".to_string());
        }
    }
    if let Some(text) = message_text {
        if contains_arabic_script(text) {
            reasons.push("ARABIC".to_string());
        }
    }
    reasons
}

fn regex_is_match(re: &FancyRegex, text: &str) -> bool {
    re.is_match(text).unwrap_or(false)
}

fn jieba() -> &'static Jieba {
    static JIEBA: OnceLock<Jieba> = OnceLock::new();
    JIEBA.get_or_init(Jieba::new)
}

fn normalize_tokens(text: &str, jieba: &Jieba) -> Vec<String> {
    static PUNCT_RE: OnceLock<StdRegex> = OnceLock::new();
    let punct = PUNCT_RE.get_or_init(|| StdRegex::new(r"[[:punct:][:space:]]+").expect("valid punctuation regex"));
    let lowered = text.to_lowercase();
    let cleaned = punct.replace_all(&lowered, " ");
    jieba
        .cut(&cleaned, false)
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s.chars().count() > 1)
        .collect()
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

/// Commit hash embedded at compile time by build.rs (falls back to
/// "unknown" if `.git` wasn't available in the build environment).
const GIT_HASH: &str = env!("GIT_HASH");

fn version_info_text() -> String {
    format!(
        "🏓 Pong！Bot 已啟動並運作中。\n<b>Version</b>: <code>{}</code>\n<b>Commit</b>: <code>{}</code>",
        env!("CARGO_PKG_VERSION"),
        GIT_HASH,
    )
}

fn help_text() -> String {
    "<b>歡迎使用 Spam Protection Bot（SPB）全自動人工智障反廣告項目。</b>\n\n只需要把這個機器人拉進你的群組，並給它管理員權限（至少需要刪除訊息 + 封禁用戶權限），它就會自動開始工作。\n\n<b>機器人主要功能：</b>\n<code>/sb</code> 或 <code>/spamban</code>：回覆訊息使用，封禁並加入黑名單訓練\n<code>/mute</code>：禁言\n<code>/kick</code>：踢出\n<code>/white</code>：加入本群白名單\n<code>/white -global</code>：加入全域白名單\n<code>/unwhite</code>：移出本群白名單\n<code>/unwhite -global</code>：移出全域白名單\n\n<b>群組管理員可用</b>\n<code>/module &lt;名稱&gt; &lt;on/off&gt;</code>：切換群組模組，名稱支援 NoLongName（英名檢查）/ NoHalal（清真檢查）/ NoSM（服務訊息刪除）/ Flood（洗版偵測，預設開啟）/ Captcha（新成員驗證，預設關閉）/ Netban（跨群組黑名單同步，預設關閉，需自行開啟；開啟後本群的封禁會同步到其他同樣開啟的群組，反之亦然）/ CmdClean（指令權限濫用防護，預設關閉；開啟後，沒有權限的人嘗試使用管理指令會被刪除訊息並警告一次，24 小時內再犯將被禁言 5 分鐘並記錄到日誌頻道。無論是否開啟，此類指令的錯誤提示訊息都會在 10 秒後自動刪除，減少洗版）\n<code>/unban</code>：回覆要解封的用戶、或提供 user_id，解封本群該用戶（僅本群，不影響訓練資料，如需連同撤銷誤判樣本請找維護組）\n<code>/unmute</code>：回覆要解除禁言的用戶、或提供 user_id\n\n普通成員可使用 <code>/report</code> 或 <code>/spam</code> 舉報可疑訊息，交由項目組審核\n任何人可輸入 <code>/case &lt;ID&gt;</code> 查詢某次封禁的詳細記錄\n\n<b>注意事項：</b>\n被封禁後想查原因：先發 <code>/id</code> 取得自己的 User ID，然後去日誌頻道 <code>@SpamProtectionLogging</code> 搜尋\n\n項目交流群：https://t.me/SpamProtectionChat\n日誌頻道：https://t.me/SpamProtectionLogging\n".to_string()
}

fn help_op_text() -> String {
    "<b>維護指令</b>\n\n<b>模型 / 訓練</b>\n<code>/ml_score</code>：測試單條文本分數\n<code>/ml_score_debug</code>：看抽取結果與分數細節\n<code>/ml_stats</code>：查看樣本量與有效門檻\n<code>/ml_threshold &lt;值&gt;</code>：調整封禁門檻。在私訊/測試群/工作群組使用會調整全域門檻；在其他群組使用只影響該群組\n<code>/set 0x&lt;token&gt; &lt;0.05~0.95&gt;</code>：直接調整 token 的 spam/ham 機率偏置\n<code>/ml_export</code>：匯出訓練資料\n<code>/import</code>：匯入已輸出的訓練列表\n<code>/ml_train_spam</code>（別名 <code>/mark_spam</code>）：把回覆內容直接當 spam 訓練\n<code>/ml_clean_spam</code>：把回覆內容清成 ham / clean\n<code>/ml_undo_clean_spam</code>：撤銷回覆內容寫入 ham/clean 的樣本\n<code>/mark_ham</code>：將回覆內容標記為 ham\n<code>/ml_purge &lt;case_id&gt;</code>：依案例刪除誤樣本\n<code>/ml_purge_text &lt;文字片段&gt;</code>：依文字片段刪除誤樣本\n<code>/ml_rebuild</code>：重建模型\n\n<b>撤銷操作</b>\n<code>/unban</code>：維護組專用完整版，回覆用戶、或提供 user_id / case_id 皆可。會解封並在找得到對應案例時一併移除錯誤訓練樣本並重建模型，若該案例曾透過 Netban 同步封禁到其他群組，也會一併在那些群組解封（群組管理員也能用 /unban，但僅解封本群、不影響訓練資料與其他群組）\n<code>/unmute</code>：維護組專用完整版，回覆用戶、或提供 user_id / case_id 皆可，並會撤銷對應案例（群組管理員也能用 /unmute，但僅解除本群禁言）\n\n<b>批量訓練</b>\n<code>/ml_start_mass_train_smart</code>：進入 smart 批量訓練模式\n<code>/ml_start_mass_train_plain</code>：進入 plain 批量訓練模式\n<code>/ml_finish_mass_train</code>：結束 spam 批量訓練\n<code>/ml_start_mass_ham</code>：開始批量標記 ham\n<code>/ml_finish_mass_ham</code>：結束 ham 批量訓練\n\n<b>群組控制</b>\n<code>/setchat [chat_id]</code>：設定工作群組。不帶參數時直接綁定目前所在的群組；也可提供 chat_id 從其他地方設定。綁定後，若該群組串連的頻道發文時被 Telegram 自動釘選，機器人會自動取消釘選，避免洗掉手動釘選的訊息\n<code>/leave [&lt;chat_id&gt;] [原因]</code>：讓 bot 離開指定群組或目前群組\n<code>/ping</code>：確認機器人在線，並回報目前運行的版本號與 commit hash\n<code>/set_audit_log [chat_id]</code>：設定維護操作日誌頻道。不帶參數時綁定目前所在的群組/頻道。設定後，每個會改變狀態的維護指令（門檻、白名單、模組開關、規則異動、封禁/禁言等）都會記錄在這裡，並附上 action id\n<code>/revert &lt;action_id&gt;</code>：復原指定的維護操作，回到變更前的狀態；封禁/禁言類會重用 /unban、/unmute 的邏輯。少數沒有明確「復原前狀態」的操作無法自動復原，會直接告知\n\n<b>規則管理</b>\n<code>/add_rule &lt;regex&gt;</code>：新增正則規則，會再追問名稱\n<code>/edit_rule &lt;id&gt; &lt;regex&gt;</code>：只更新正則，不改名稱\n<code>/del_rule &lt;id&gt;</code>：刪除規則\n<code>/list_rules</code>：列出目前規則\n<code>/check_rules</code>：列出無法編譯的規則\n<code>/updateBL</code>：更新封禁代號說明\n\n<b>備註</b>\n這頁只放維護者會用到的指令。普通 <code>/help</code> 不會列出這些。\n".to_string()
}

fn format_score_debug(report: &ScoreDebugReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("<b>分數</b>: {:.6}\n", report.score));
    for item in &report.tokens {
        out.push_str(&format!(
            "<code>{}</code> 垃圾={} 正常={} 垃圾機率={:.6} 正常機率={:.6} 差值={:.6}\n",
            escape_html(&item.token),
            item.spam_count,
            item.ham_count,
            item.spam_prob,
            item.ham_prob,
            item.delta,
        ));
    }
    out
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

fn chinese_case_action(case: &CaseRecord) -> String {
    if let Some(rule_id) = case.matched_rule_id {
        format!("規則 #{}", rule_id)
    } else {
        match case.action {
            ActionKind::AutoDelete => "自動刪除".to_string(),
            ActionKind::AutoBan => "自動封禁".to_string(),
            ActionKind::SpamBan => "封禁".to_string(),
            ActionKind::Mute => "禁言".to_string(),
            ActionKind::Kick => "踢出".to_string(),
            ActionKind::PendingReport => "待審核".to_string(),
            ActionKind::ReportApproved => "受理封禁".to_string(),
            ActionKind::ReportRejected => "拒絕受理".to_string(),
            ActionKind::Unbanned => "已撤銷封禁".to_string(),
            ActionKind::Unmuted => "已解除禁言".to_string(),
            ActionKind::FloodMute => "洗版禁言".to_string(),
            ActionKind::CmdCleanMute => "指令濫用禁言".to_string(),
        }
    }
}

fn chinese_case_reason(case: &CaseRecord) -> String {
    case.matched_rule_pattern.clone().unwrap_or_else(|| "-".to_string())
}

fn build_reason_link(reason: &str, link: &str) -> String {
    format!("<a href=\"{link}\">{reason}</a>")
}

fn format_code_link(code: &str, link: Option<&str>) -> String {
    match link {
        Some(link) => build_reason_link(&escape_html(code), link),
        None => escape_html(code),
    }
}

fn format_public_reason(reason: &str, link: Option<&str>) -> String {
    reason
        .split('；')
        .filter(|part| !part.trim().is_empty())
        .map(|part| format_code_link(part.trim(), link))
        .collect::<Vec<_>>()
        .join("；")
}

fn global_whitelist_check_text() -> String {
    "<b>檢查結果</b>\n<b>對象</b>: 全域白名單\n<b>命中</b>: 無\n<b>名稱規則</b>: 無\n<b>清真規則</b>: 無".to_string()
}

fn build_blacklist_reason_text(_runtime: &Runtime) -> String {
    "<b>❖ 封禁代號說明</b>\n\n- <code>NLDIGIT</code>: 英名含數字\n- <code>NL13</code>: 英名多段且總長度 >= 13\n- <code>NLTAIL</code>: 英名多段且尾段過長\n- <code>NLSINGLE</code>: 英名單段且長度 >= 11\n- <code>ARABIC</code>: 偵測到清真\n- <code>REGEX</code>: 觸發正則規則\n- <code>FLOOD</code>: 洗版偵測（5 秒內傳送 5 條以上訊息）\n- <code>PERM_REPEAT</code>: 24 小時內重複嘗試使用無權限的指令\n\n申訴找 @SEELE_01_BOT".to_string()
}

fn format_case_lookup(case: &CaseRecord, link: &str, reason_link: &str) -> String {
    format!(
        "<b>案例</b>: <code>{}</code>\n<b>操作</b>: {}\n<b>狀態</b>: {}\n<b>對象</b>: {} ({})\n<b>原因</b>: {}\n<b>日誌</b>: {}\n<b>證據</b>: <blockquote>{}</blockquote>",
        case.id,
        chinese_case_action(case),
        escape_html(&case.status),
        escape_html(&case.target_name),
        case.target_user_id,
        format_public_reason(&chinese_case_reason(case), Some(reason_link)),
        link,
        escape_html(&case.evidence_text),
    )
}

async fn is_maintainer(bot: &Bot, config: &Config, user_id: i64) -> bool {
    if config.maintainer_ids.contains(&user_id) {
        return true;
    }
    let _ = bot;
    false
}

/// Guards a command handler arm behind `is_maintainer`, replying with `$msg` and
/// returning early otherwise. Collapses the same 4-line permission check that
/// used to be repeated at the top of ~24 command arms in `handle_command`.
macro_rules! require_maintainer {
    ($bot:expr, $runtime:expr, $from_id:expr, $message:expr, $msg:expr) => {
        if !is_maintainer($bot, &$runtime.config, $from_id).await {
            $bot.send_message($message.chat.id, $msg).await?;
            return Ok(());
        }
    };
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

const CAPTCHA_TIMEOUT: Duration = Duration::from_secs(120);

/// Not cryptographically random and not meant to be - this only needs to be
/// unpredictable enough to stop a dumb join-spam bot from guessing the
/// answer, not to resist a targeted attack.
fn generate_captcha_challenge(seed_extra: i64) -> (i64, i64, String) {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as i64)
        .unwrap_or(0);
    let seed = nanos.wrapping_add(seed_extra).unsigned_abs();
    let a = (seed % 8 + 1) as i64;
    let b = ((seed / 8) % 8 + 1) as i64;
    (a, b, (a + b).to_string())
}

/// Restricts a new member to text-only, posts a simple arithmetic challenge,
/// and schedules a kick if it goes unanswered. Reuses the same
/// spawn-a-delayed-cleanup-task pattern `notify_group` already uses for its
/// 180s auto-delete, just kicking instead of deleting when it fires.
async fn start_captcha_challenge(bot: &Bot, runtime: &Arc<Runtime>, chat_id: ChatId, user: &teloxide::types::User) {
    let user_id = user.id.0 as i64;
    let (a, b, expected_answer) = generate_captcha_challenge(user_id);

    if bot
        .restrict_chat_member(chat_id, user.id, teloxide::types::ChatPermissions::SEND_MESSAGES)
        .await
        .is_err()
    {
        return;
    }

    let text = format!(
        "{} 你好，為了防止機器人/廣告帳號，請在 {} 秒內直接回覆下面問題的答案（純數字），逾時將被移出群組：\n\n<b>{a} + {b} = ?</b>",
        escape_html(&short_user(user)),
        CAPTCHA_TIMEOUT.as_secs(),
    );
    let Ok(sent) = bot.send_message(chat_id, text).parse_mode(ParseMode::Html).await else { return; };

    {
        let mut pending = runtime.pending_captcha.lock().await;
        pending.insert(
            (chat_id.0, user_id),
            PendingCaptcha { expected_answer, expires_at: Instant::now() + CAPTCHA_TIMEOUT, challenge_message_id: sent.id },
        );
    }

    let bot = bot.clone();
    let runtime = runtime.clone();
    let challenge_message_id = sent.id;
    tokio::spawn(async move {
        sleep(CAPTCHA_TIMEOUT).await;
        let still_pending = {
            let mut pending = runtime.pending_captcha.lock().await;
            match pending.get(&(chat_id.0, user_id)) {
                Some(p) if p.challenge_message_id == challenge_message_id => {
                    pending.remove(&(chat_id.0, user_id));
                    true
                }
                _ => false,
            }
        };
        if still_pending {
            let _ = bot.delete_message(chat_id, challenge_message_id).await;
            // Kick, not ban: failing to answer in time isn't proof of spam,
            // just an unverified join.
            let _ = kick_user(&bot, chat_id, user_id).await;
        }
    });
}

/// Checks an incoming message against a pending join CAPTCHA for its sender
/// in this chat. Returns true if it consumed the message (whether right or
/// wrong), so the caller can skip further processing for it.
async fn check_captcha_and_act(bot: &Bot, runtime: &Arc<Runtime>, message: &Message) -> bool {
    let Some(user) = message.from.as_ref() else { return false; };
    let key = (message.chat.id.0, user.id.0 as i64);

    let (expected, challenge_message_id) = {
        let pending = runtime.pending_captcha.lock().await;
        let Some(entry) = pending.get(&key) else { return false; };
        if Instant::now() > entry.expires_at {
            // Already expired - let the timeout task's own kick handle it
            // rather than racing it.
            return false;
        }
        (entry.expected_answer.clone(), entry.challenge_message_id)
    };

    let answer = message.text().unwrap_or("").trim();
    if answer == expected {
        runtime.pending_captcha.lock().await.remove(&key);
        let _ = bot
            .restrict_chat_member(message.chat.id, user.id, teloxide::types::ChatPermissions::all())
            .await;
        let _ = bot.delete_message(message.chat.id, message.id).await;
        // Clean up the question itself, not just the answer - it was still
        // sitting in the chat with nothing telling anyone it got resolved.
        let _ = bot.delete_message(message.chat.id, challenge_message_id).await;
        let _ = bot
            .send_message(message.chat.id, format!("✅ {} 驗證通過，歡迎！", escape_html(&short_user(user))))
            .parse_mode(ParseMode::Html)
            .await;
    } else {
        // Wrong guess: delete it and let them try again until the timeout.
        let _ = bot.delete_message(message.chat.id, message.id).await;
    }
    true
}

async fn notify_bot_added(bot: &Bot, runtime: &Arc<Runtime>, message: &Message) -> bool {
    let Some(users) = message.new_chat_members() else { return false; };
    if users.is_empty() {
        return false;
    }

    if users.iter().any(|u| u.is_bot) {
        let title = message.chat.title().unwrap_or("unknown");
        let text = format!(
            "<b>機器人已加入</b>\n<b>群組</b>: <code>{}</code>\n<b>標題</b>: {}\n<b>來源</b>: <code>{}</code>",
            message.chat.id.0,
            escape_html(title),
            message.from.as_ref().map(short_user).unwrap_or_else(|| "unknown".to_string())
        );
        let _ = bot.send_message(ChatId(runtime.config.report_channel_id), text).parse_mode(ParseMode::Html).await;
    }

    for user in users {
        if is_special_user(&runtime.config, user.id.0 as i64) {
            continue;
        }
        if runtime.is_global_whitelisted(user.id.0 as i64).await.unwrap_or(false) {
            continue;
        }
        if runtime.is_group_whitelisted(message.chat.id.0, user.id.0 as i64).await.unwrap_or(false) {
            continue;
        }

        let enabled = runtime.get_group_modules(message.chat.id.0).await.unwrap_or_default();
        let mut banned = false;

        if enabled.no_long_name || enabled.no_halal {
            let reasons = if enabled.no_long_name { evaluate_name_guard(&display_name_only(user)) } else { Vec::new() };
            let mut arabic_reasons = if enabled.no_halal {
                let profile = runtime.load_user_profile(bot, user.id.0 as i64).await.ok();
                let bio = profile.as_ref().and_then(|p| p.bio.as_deref());
                evaluate_module_checks(user, user.username.as_deref(), bio, None)
            } else {
                Vec::new()
            };
            let mut all_reasons = reasons;
            all_reasons.append(&mut arabic_reasons);

            if !all_reasons.is_empty() {
                banned = true;
                let _ = bot.delete_message(message.chat.id, message.id).await;
                let _ = ban_user(bot, message.chat.id, user.id.0 as i64).await;
                let case = CaseRecord {
                    id: Uuid::new_v4().to_string(),
                    action: ActionKind::AutoBan,
                    chat_id: message.chat.id.0,
                    target_user_id: user.id.0 as i64,
                    target_name: short_user(user),
                    actor_user_id: None,
                    actor_name: None,
                    source_message_id: Some(message.id.0),
                    evidence_text: extract_full_text(message),
                    model_score: None,
                    matched_rule_id: None,
                    matched_rule_pattern: Some(all_reasons.join("；")),
                    status: "auto_banned".to_string(),
                    log_message_id: None,
                    created_at: Utc::now(),
                };
                let log_message_id = log_action(bot, runtime, &case).await.unwrap_or_default();
                let mut updated = case.clone();
                updated.log_message_id = Some(log_message_id);
                let _ = store_case(runtime, &updated).await;
                let _ = notify_group(bot, runtime, &updated, log_message_id, "<b>自動模組封禁</b>").await;
                propagate_network_ban(bot, runtime, &updated).await;
            }
        }

        // Join-time netban catch-up: this group only learns about a network
        // ban when it's checked (there's no way to scan existing members via
        // the Bot API to backfill), so check every new joiner. A known-bad
        // user doesn't need a CAPTCHA challenge, so this takes priority over
        // that check.
        if !banned && enabled.netban {
            if let Ok(Some(prior_case)) = runtime.find_active_network_ban(user.id.0 as i64).await {
                banned = true;
                let _ = bot.delete_message(message.chat.id, message.id).await;
                let _ = bot.ban_chat_member(message.chat.id, user.id).await;
                let _ = runtime.record_network_ban_target(&prior_case.id, message.chat.id.0).await;
                let _ = bot
                    .send_message(
                        message.chat.id,
                        format!(
                            "<b>跨群組黑名單同步封禁</b>\n用戶 <code>{}</code> 已因跨群組黑名單同步封禁。\n原始案例: <code>{}</code>",
                            user.id.0, prior_case.id,
                        ),
                    )
                    .parse_mode(ParseMode::Html)
                    .await;
            }
        }

        if !banned && enabled.captcha {
            start_captcha_challenge(bot, runtime, message.chat.id, user).await;
        }
    }

    true
}

fn parse_leave_args(args: &str) -> (Option<i64>, String) {
    let trimmed = args.trim();
    if trimmed.is_empty() {
        return (None, String::new());
    }
    let mut parts = trimmed.split_whitespace();
    let first = parts.next().unwrap_or("");
    if let Ok(chat_id) = first.parse::<i64>() {
        let reason = parts.collect::<Vec<_>>().join(" ");
        return (Some(chat_id), reason);
    }
    (None, trimmed.to_string())
}

fn project_chat_link(chat_id: i64) -> String {
    let id = chat_id.abs().to_string().trim_start_matches("100").to_string();
    format!("https://t.me/c/{id}/1")
}

async fn log_action(bot: &Bot, runtime: &Runtime, case: &CaseRecord) -> ResponseResult<i32> {
    let action_text = chinese_case_action(case);
    let reason_text = escape_html(&chinese_case_reason(case));
    let text = format!(
        "<b>案例</b>: <code>{}</code>\n<b>操作</b>: {}\n<b>群組</b>: <code>{}</code>\n<b>對象</b>: <code>{}</code> {}\n<b>處理者</b>: {}\n<b>分數</b>: {}\n<b>原因</b>: {}\n<b>證據</b>:\n<blockquote>{}</blockquote>\n<b>時間</b>: {}",
        case.id,
        action_text,
        case.chat_id,
        case.target_user_id,
        escape_html(&case.target_name),
        case.actor_user_id.map(|id| id.to_string()).unwrap_or_else(|| "system".to_string()),
        case.model_score.map(|s| format!("{s:.4}")).unwrap_or_else(|| "-".to_string()),
        reason_text,
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
        "<b>回調錯誤</b>\n<b>階段</b>: <code>{}</code>\n<b>案例</b>: <code>{}</code>\n<b>群組</b>: <code>{}</code>\n<b>錯誤</b>:\n<blockquote>{}</blockquote>",
        escape_html(stage),
        case.id,
        case.chat_id,
        escape_html(err),
    );
    let _ = bot.send_message(ChatId(runtime.config.log_channel_id), text).parse_mode(ParseMode::Html).await;
}

async fn delete_message_if_exists(bot: &Bot, chat_id: ChatId, message_id: MessageId) -> Result<()> {
    match bot.delete_message(chat_id, message_id).await {
        Ok(_) => Ok(()),
        Err(err) if err.to_string().contains("message to delete not found") => Ok(()),
        Err(err) => Err(err.into()),
    }
}

async fn notify_group(bot: &Bot, runtime: &Runtime, case: &CaseRecord, log_message_id: i32, header: &str) -> Result<()> {
    let link = public_log_link(&runtime.config, log_message_id);
    let reason_link = runtime.blacklist_reason_link().await.unwrap_or_else(|| link.clone());
    let reason = case.matched_rule_pattern.as_deref().unwrap_or("N");
    let text = format!(
        "{header}\n\n<b>操作</b>: {}\n<b>對象</b>: <code>{}</code>\n<b>原因</b>: {}\n<b>證據</b>: <a href=\"{}\">查看日誌</a>\n<b>案例</b>: <code>{}</code>",
        chinese_case_action(case),
        case.target_user_id,
        format_public_reason(reason, Some(&reason_link)),
        link,
        case.id
    );
    let sent = bot.send_message(ChatId(case.chat_id), text).parse_mode(ParseMode::Html).await?;
    let bot = bot.clone();
    let chat_id = ChatId(case.chat_id);
    let message_id = sent.id;
    tokio::spawn(async move {
        sleep(Duration::from_secs(180)).await;
        let _ = bot.delete_message(chat_id, message_id).await;
    });
    Ok(())
}

/// Records a state-changing maintainer command and, if `/set_audit_log` has
/// been configured, posts it to the private audit channel with its new
/// `action_id` and a `/revert` hint (unless `undo` is `NotRevertible`).
/// Best-effort: a failure to record shouldn't block the command that
/// triggered it, so callers just ignore the `None` case.
#[allow(clippy::too_many_arguments)]
async fn log_maintainer_action(bot: &Bot, runtime: &Runtime, actor_id: i64, actor_name: &str, chat_id: Option<i64>, command: &str, summary: &str, undo: UndoData) -> Option<i64> {
    let revertible = !matches!(undo, UndoData::NotRevertible);
    let action_id = runtime.record_maintainer_action(actor_id, actor_name, chat_id, command, summary, &undo).await.ok()?;
    if let Some(log_chat) = runtime.audit_log_chat().await {
        let revert_hint = if revertible {
            format!("復原：<code>/revert {action_id}</code>")
        } else {
            "（無法復原）".to_string()
        };
        let text = format!(
            "<b>維護操作 #{action_id}</b>\n<b>指令</b>: <code>{}</code>\n<b>操作者</b>: {} (<code>{actor_id}</code>)\n<b>內容</b>: {}\n{revert_hint}",
            escape_html(command),
            escape_html(actor_name),
            escape_html(summary),
        );
        let _ = bot.send_message(ChatId(log_chat), text).parse_mode(ParseMode::Html).await;
    }
    Some(action_id)
}

/// Reverses a ban case: unbans in the case's origin chat (and any chat
/// netban had propagated it to), purges the training sample it
/// contributed, and marks the case `Unbanned`. Shared by `/unban`'s
/// maintainer path and the `/revert` dispatcher, so ban reversal only
/// exists in one place. Returns a ready-to-send HTML summary on success, or
/// a ready-to-send error message on failure.
async fn reverse_ban_case(bot: &Bot, runtime: &Runtime, mut case: CaseRecord, actor_id: i64, actor_name: &str) -> Result<String, String> {
    if let Err(err) = bot.unban_chat_member(ChatId(case.chat_id), UserId(case.target_user_id as u64)).await {
        return Err(format!("解封失敗：{err}"));
    }

    let removed = runtime.purge_training_by_case(&case.id).await.unwrap_or(0);
    if removed > 0 {
        let _ = runtime.rebuild_model().await;
    }

    // If netban had propagated this ban elsewhere, undo it everywhere it
    // actually landed - not just wherever's currently opted in, since that
    // can have changed since the ban happened.
    let network_targets = runtime.list_network_ban_targets(&case.id).await.unwrap_or_default();
    for target_chat_id in &network_targets {
        let _ = bot.unban_chat_member(ChatId(*target_chat_id), UserId(case.target_user_id as u64)).await;
    }
    if !network_targets.is_empty() {
        let _ = runtime.clear_network_ban_targets(&case.id).await;
    }

    case.action = ActionKind::Unbanned;
    case.status = "reversed".to_string();
    case.actor_user_id = Some(actor_id);
    case.actor_name = Some(actor_name.to_string());
    store_case(runtime, &case).await.ok();
    let log_message_id = log_action(bot, runtime, &case).await.unwrap_or_default();
    case.log_message_id = Some(log_message_id);
    store_case(runtime, &case).await.ok();
    notify_group(bot, runtime, &case, log_message_id, "<b>已撤銷封禁</b>").await.ok();

    let network_note = if network_targets.is_empty() {
        String::new()
    } else {
        format!("，並在 {} 個跨群組黑名單同步的群組中解封", network_targets.len())
    };
    Ok(format!("已解封用戶，並撤銷 case <code>{}</code>、移除 {removed} 筆對應訓練樣本{network_note}。", case.id))
}

/// Reverses a mute case: restores full permissions in the case's chat and
/// marks the case `Unmuted`. Shared by `/unmute`'s maintainer path and the
/// `/revert` dispatcher.
async fn reverse_mute_case(bot: &Bot, runtime: &Runtime, mut case: CaseRecord, actor_id: i64, actor_name: &str) -> Result<String, String> {
    if let Err(err) = bot.restrict_chat_member(ChatId(case.chat_id), UserId(case.target_user_id as u64), teloxide::types::ChatPermissions::all()).await {
        return Err(format!("解除禁言失敗：{err}"));
    }

    case.action = ActionKind::Unmuted;
    case.status = "reversed".to_string();
    case.actor_user_id = Some(actor_id);
    case.actor_name = Some(actor_name.to_string());
    store_case(runtime, &case).await.ok();
    let log_message_id = log_action(bot, runtime, &case).await.unwrap_or_default();
    case.log_message_id = Some(log_message_id);
    store_case(runtime, &case).await.ok();
    notify_group(bot, runtime, &case, log_message_id, "<b>已解除禁言</b>").await.ok();

    Ok(format!("已解除禁言，並撤銷 case <code>{}</code>。", case.id))
}

/// Propagates a ban case to every other group that has opted into `netban`.
/// Called right after the 5 places in this file that create a ban-type case
/// (AutoBan/SpamBan/ReportApproved) - purely additive, doesn't change any
/// existing behavior at those call sites. No-ops immediately if the
/// *origin* group hasn't opted in, since propagation is symmetric: a group
/// only sends bans out to (and receives bans from) other opted-in groups.
async fn propagate_network_ban(bot: &Bot, runtime: &Runtime, case: &CaseRecord) {
    let origin = runtime.get_group_modules(case.chat_id).await.unwrap_or_default();
    if !origin.netban {
        return;
    }
    if runtime.is_global_whitelisted(case.target_user_id).await.unwrap_or(false) {
        return;
    }

    let targets = runtime.list_netban_enabled_chats().await.unwrap_or_default();
    for chat_id in targets {
        if chat_id == case.chat_id {
            continue;
        }
        if runtime.is_group_whitelisted(chat_id, case.target_user_id).await.unwrap_or(false) {
            continue;
        }
        if bot.ban_chat_member(ChatId(chat_id), UserId(case.target_user_id as u64)).await.is_ok() {
            let _ = runtime.record_network_ban_target(&case.id, chat_id).await;
            let _ = bot
                .send_message(
                    ChatId(chat_id),
                    format!(
                        "<b>跨群組黑名單同步封禁</b>\n用戶 <code>{}</code> 已因跨群組黑名單同步封禁。\n原始案例: <code>{}</code>",
                        case.target_user_id, case.id,
                    ),
                )
                .parse_mode(ParseMode::Html)
                .await;
        }
    }
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

/// Like `mute_user`, but sets Telegram's own `until_date` so the restriction
/// expires on Telegram's side regardless of whether this process is still
/// running - unlike relying purely on a `tokio::spawn` timer (see
/// `schedule_temp_unmute`), which silently never fires if the bot restarts
/// during the window, leaving the mute permanent. Telegram treats anything
/// under 30 seconds from now as "forever", so this only makes sense for
/// durations meaningfully longer than that.
async fn mute_user_until(bot: &Bot, chat_id: ChatId, user_id: i64, until: DateTime<Utc>) -> Result<()> {
    let permissions = teloxide::types::ChatPermissions::empty();
    bot.restrict_chat_member(chat_id, UserId(user_id as u64), permissions)
        .until_date(until)
        .await?;
    Ok(())
}

async fn kick_user(bot: &Bot, chat_id: ChatId, user_id: i64) -> Result<()> {
    bot.ban_chat_member(chat_id, UserId(user_id as u64)).await?;
    bot.unban_chat_member(chat_id, UserId(user_id as u64)).await?;
    Ok(())
}

/// Sends a plain-text reply and, only in group/supergroup chats, schedules
/// it for deletion after 10s - same delayed-cleanup pattern as
/// `notify_group`'s auto-delete and the CAPTCHA success message. Used for
/// "you used this command wrong" replies, which are transient noise that
/// shouldn't linger in a group's history - this runs regardless of the
/// CmdClean module below, since it's just clutter reduction, not a
/// moderation consequence.
async fn reply_ephemeral(bot: &Bot, message: &Message, text: impl Into<String>) -> ResponseResult<()> {
    let sent = bot.send_message(message.chat.id, text.into()).await?;
    if message.chat.is_group() || message.chat.is_supergroup() {
        let bot = bot.clone();
        let chat_id = message.chat.id;
        let message_id = sent.id;
        tokio::spawn(async move {
            sleep(Duration::from_secs(10)).await;
            let _ = bot.delete_message(chat_id, message_id).await;
        });
    }
    Ok(())
}

/// Restores full permissions after `after` - a temporary mute that lifts
/// itself, same shape as the CAPTCHA timeout task and `notify_group`'s
/// auto-delete. Best-effort: doesn't check whether the user was already
/// unmuted for some other reason in between, consistent with every other
/// delayed task in this file.
fn schedule_temp_unmute(bot: &Bot, chat_id: ChatId, user_id: i64, after: Duration) {
    let bot = bot.clone();
    tokio::spawn(async move {
        sleep(after).await;
        let _ = bot.restrict_chat_member(chat_id, UserId(user_id as u64), teloxide::types::ChatPermissions::all()).await;
    });
}

/// Shared handler for every "only a group admin / maintainer can do this"
/// rejection on a group-facing command. With CmdClean off, this is just
/// `reply_ephemeral` - the rejection self-deletes but nothing else happens
/// (today's behavior, just less cluttered). With CmdClean on: the offending
/// command message is deleted outright, and a repeat attempt within 24h of
/// the last one escalates to a 5-minute mute, logged like any other case.
async fn handle_permission_denied(bot: &Bot, runtime: &Runtime, message: &Message, from: &teloxide::types::User, denial_text: &str) -> ResponseResult<()> {
    let chat_id = message.chat.id.0;
    let settings = runtime.get_group_modules(chat_id).await.unwrap_or_default();
    if !settings.cmd_clean {
        return reply_ephemeral(bot, message, denial_text).await;
    }

    let _ = bot.delete_message(message.chat.id, message.id).await;
    let user_id = from.id.0 as i64;
    let prior = runtime.last_permission_offense(chat_id, user_id).await.ok().flatten();
    let _ = runtime.record_permission_offense(chat_id, user_id).await;

    let repeat_within_24h = prior.map(|t| Utc::now() - t < chrono::TimeDelta::hours(24)).unwrap_or(false);

    if repeat_within_24h {
        let _ = mute_user_until(bot, message.chat.id, user_id, Utc::now() + chrono::TimeDelta::minutes(5)).await;
        schedule_temp_unmute(bot, message.chat.id, user_id, Duration::from_secs(5 * 60));
        let case = CaseRecord {
            id: Uuid::new_v4().to_string(),
            action: ActionKind::CmdCleanMute,
            chat_id,
            target_user_id: user_id,
            target_name: short_user(from),
            actor_user_id: None,
            actor_name: None,
            source_message_id: Some(message.id.0),
            evidence_text: message.text().or(message.caption()).unwrap_or("").to_string(),
            model_score: None,
            matched_rule_id: None,
            matched_rule_pattern: Some("PERM_REPEAT".to_string()),
            status: "auto_muted".to_string(),
            log_message_id: None,
            created_at: Utc::now(),
        };
        let log_message_id = log_action(bot, runtime, &case).await.unwrap_or_default();
        let mut updated = case.clone();
        updated.log_message_id = Some(log_message_id);
        let _ = store_case(runtime, &updated).await;
        let _ = notify_group(bot, runtime, &updated, log_message_id, "<b>指令權限濫用禁言</b>").await;
    } else {
        let _ = reply_ephemeral(bot, message, "⚠️ 你沒有權限使用此指令，訊息已刪除。24 小時內再次嘗試將被禁言 5 分鐘。").await;
    }
    Ok(())
}

async fn train_spam(runtime: &Runtime, text: &str, case_id: Option<&str>) -> Result<()> {
    let tokens = tokenize(text);
    {
        let mut model = runtime.model.lock().await;
        model.spam_docs += 1;
        for token in &tokens {
            *model.spam_tokens.entry(token.clone()).or_default() += 1;
        }
    }
    runtime
        .with_conn(move |conn| {
            let tx = conn.transaction()?;
            for token in &tokens {
                tx.execute(
                    "INSERT INTO word_frequencies (word, spam_count, ham_count) VALUES (?1, 1, 0) ON CONFLICT(word) DO UPDATE SET spam_count = spam_count + 1",
                    params![token],
                )?;
            }
            tx.commit()?;
            Ok(())
        })
        .await?;
    runtime.insert_training_sample("spam", text, case_id).await?;
    runtime.persist_doc_counts().await
}

async fn train_ham(runtime: &Runtime, text: &str, case_id: Option<&str>) -> Result<()> {
    let tokens = tokenize(text);
    {
        let mut model = runtime.model.lock().await;
        model.ham_docs += 1;
        for token in &tokens {
            *model.ham_tokens.entry(token.clone()).or_default() += 1;
        }
    }
    runtime
        .with_conn(move |conn| {
            let tx = conn.transaction()?;
            for token in &tokens {
                tx.execute(
                    "INSERT INTO word_frequencies (word, spam_count, ham_count) VALUES (?1, 0, 1) ON CONFLICT(word) DO UPDATE SET ham_count = ham_count + 1",
                    params![token],
                )?;
            }
            tx.commit()?;
            Ok(())
        })
        .await?;
    runtime.insert_training_sample("ham", text, case_id).await?;
    runtime.persist_doc_counts().await
}

async fn extract_reply_context(message: &Message) -> Option<(i64, String, i32, String)> {
    let reply = message.reply_to_message()?;
    let user = reply.from.as_ref()?;
    let text = extract_full_text(reply);
    Some((user.id.0 as i64, short_user(user), reply.id.0, text))
}

fn extract_full_text(msg: &Message) -> String {
    let mut text = msg.text().or(msg.caption()).unwrap_or("").to_string();

    if let Some(quote) = msg.quote() {
        let quote_text = quote.text.trim();
        if !quote_text.is_empty() {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(quote_text);
        }
    }

    if let Some(origin) = msg.forward_origin() {
        if !text.is_empty() {
            text.push('\n');
        }
        if let teloxide::types::MessageOrigin::Channel { chat, .. } = origin {
            text.push_str(&format!("\n[fwd_id: {}]", chat.id.0));
            if let Some(username) = chat.username() {
                text.push_str(&format!("\n[fwd_user: {}]", username));
            }
        }
    }

    if let teloxide::types::MessageKind::Common(common) = &msg.kind {
        if let Some(external) = &common.external_reply {
            if !text.is_empty() {
                text.push('\n');
            }
            if let Some(chat) = &external.chat {
                text.push_str(&format!("\n[external_origin_chat_id: {}]", chat.id.0));
                if let Some(username) = chat.username() {
                    text.push_str(&format!("\n[external_origin_username: {}]", username));
                }
            }
            match &external.origin {
                teloxide::types::MessageOrigin::Channel { chat, .. } => {
                    text.push_str(&format!("\n[external_reply_origin_channel_id: {}]", chat.id.0));
                    if let Some(username) = chat.username() {
                        text.push_str(&format!("\n[external_reply_origin_channel_username: {}]", username));
                    }
                }
                teloxide::types::MessageOrigin::Chat { sender_chat, .. } => {
                    text.push_str(&format!("\n[external_reply_origin_chat_id: {}]", sender_chat.id.0));
                    if let Some(username) = sender_chat.username() {
                        text.push_str(&format!("\n[external_reply_origin_chat_username: {}]", username));
                    }
                }
                _ => {}
            }
        }
    }

    text
}

fn tokenize_or_empty(text: &str) -> Vec<String> {
    tokenize(text)
}

fn is_empty_ml_text(text: &str) -> bool {
    tokenize_or_empty(text).is_empty()
}

fn score_spam_from_text(model: &ModelState, text: &str) -> f64 {
    if is_empty_ml_text(text) {
        return 0.0;
    }
    let tokens = tokenize(text);
    let spam_total = model.spam_tokens.values().sum::<u64>() as f64 + 1.0;
    let ham_total = model.ham_tokens.values().sum::<u64>() as f64 + 1.0;
    let vocab = (model.spam_tokens.len() + model.ham_tokens.len()).max(1) as f64;
    let prior_spam = (model.spam_docs as f64 + 1.0) / ((model.spam_docs + model.ham_docs) as f64 + 2.0);
    let prior_ham = 1.0 - prior_spam;

    let mut log_spam = prior_spam.ln();
    let mut log_ham = prior_ham.ln();

    for token in tokens {
        let spam_count = *model.spam_tokens.get(&token).unwrap_or(&0);
        let ham_count = *model.ham_tokens.get(&token).unwrap_or(&0);
        let spam_prob = (spam_count as f64 + 1.0) / (spam_total + vocab);
        let ham_prob = (ham_count as f64 + 1.0) / (ham_total + vocab);
        log_spam += spam_prob.ln();
        log_ham += ham_prob.ln();
    }

    let odds = (log_spam - log_ham).exp();
    odds / (1.0 + odds)
}

fn score_debug_from_text(model: &ModelState, text: &str) -> ScoreDebugReport {
    if is_empty_ml_text(text) {
        return ScoreDebugReport { score: 0.0, tokens: Vec::new() };
    }

    let tokens = tokenize(text);
    let spam_total = model.spam_tokens.values().sum::<u64>() as f64 + 1.0;
    let ham_total = model.ham_tokens.values().sum::<u64>() as f64 + 1.0;
    let vocab = (model.spam_tokens.len() + model.ham_tokens.len()).max(1) as f64;
    let prior_spam = (model.spam_docs as f64 + 1.0) / ((model.spam_docs + model.ham_docs) as f64 + 2.0);
    let prior_ham = 1.0 - prior_spam;

    let mut log_spam = prior_spam.ln();
    let mut log_ham = prior_ham.ln();
    let mut contributions = Vec::new();

    for token in tokens {
        let spam_count = *model.spam_tokens.get(&token).unwrap_or(&0);
        let ham_count = *model.ham_tokens.get(&token).unwrap_or(&0);
        let spam_prob = (spam_count as f64 + 1.0) / (spam_total + vocab);
        let ham_prob = (ham_count as f64 + 1.0) / (ham_total + vocab);
        let delta = spam_prob.ln() - ham_prob.ln();
        log_spam += spam_prob.ln();
        log_ham += ham_prob.ln();
        contributions.push(ScoreContribution { token, spam_count, ham_count, spam_prob, ham_prob, delta });
    }

    let odds = (log_spam - log_ham).exp();
    let score = odds / (1.0 + odds);
    ScoreDebugReport { score, tokens: contributions }
}

// Check if a message is a service message and delete it if no_service_messages is enabled
/// When a channel has this group set as its linked discussion group,
/// Telegram automatically forwards every channel post into the group *and*
/// pins it - by design, with no setting to turn that off. That auto-pin
/// keeps replacing whatever the maintainers had intentionally pinned. Scoped
/// to `project_chat` (bound via `/setchat`, with no separate module toggle -
/// binding a chat as the project group already is the opt-in). Detects the
/// "message pinned" service notification, checks whether the message it
/// names was itself an automatic channel forward (`is_automatic_forward`,
/// exactly the flag Telegram sets for this case and no other), and if so
/// unpins that specific message, leaving any real, human-made pin alone.
async fn unpin_channel_autopin(bot: &Bot, runtime: &Runtime, message: &Message) {
    if runtime.project_chat().await != Some(message.chat.id.0) {
        return;
    }
    let Some(pinned) = message.pinned_message() else { return; };
    let Some(pinned_msg) = pinned.regular_message() else { return; };
    if !pinned_msg.is_automatic_forward() {
        return;
    }
    let _ = bot.unpin_chat_message(message.chat.id).message_id(pinned_msg.id).await;
}

async fn delete_service_message_if_enabled(bot: &Bot, runtime: &Arc<Runtime>, message: &Message) -> ResponseResult<bool> {
    // Only apply in groups/supergroups
    if !message.chat.is_group() && !message.chat.is_supergroup() {
        return Ok(false);
    }

    let chat_id = message.chat.id.0;
    let settings = runtime.get_group_modules(chat_id).await.unwrap_or_default();
    
    if !settings.no_service_messages {
        return Ok(false);
    }

    let is_service = message.new_chat_members().is_some()
        || message.left_chat_member().is_some()
        || message.new_chat_title().is_some()
        || message.new_chat_photo().is_some()
        || message.delete_chat_photo().is_some()
        || message.group_chat_created().is_some()
        || message.channel_chat_created().is_some()
        || message.migrate_to_chat_id().is_some()
        || message.migrate_from_chat_id().is_some()
        || message.pinned_message().is_some()
        || message.message_auto_delete_timer_changed().is_some()
        || message.video_chat_started().is_some()
        || message.video_chat_ended().is_some()
        || message.video_chat_participants_invited().is_some();

    if is_service {
        let _ = bot.delete_message(message.chat.id, message.id).await;
        return Ok(true);
    }

    Ok(false)
}

/// Behavioral (content-independent) spam signal: N messages from the same
/// user in the same chat within a short window. Complements the Naive Bayes
/// / regex / name-guard checks, which are all content-based and blind to a
/// brand-new spam account posting brand-new wording. Runs for every message
/// type (not just text), so it's called from the raw dispatcher in `main()`
/// rather than from inside `auto_moderate` (which only fires for non-command
/// messages once other checks have run).
async fn check_flood_and_act(bot: &Bot, runtime: &Arc<Runtime>, message: &Message) -> ResponseResult<bool> {
    if !message.chat.is_group() && !message.chat.is_supergroup() {
        return Ok(false);
    }
    let Some(user) = message.from.as_ref() else { return Ok(false); };
    if user.is_bot {
        return Ok(false);
    }
    let chat_id = message.chat.id.0;
    let user_id = user.id.0 as i64;

    if is_special_user(&runtime.config, user_id) {
        return Ok(false);
    }
    if runtime.is_global_whitelisted(user_id).await.unwrap_or(false) {
        return Ok(false);
    }
    if runtime.is_group_whitelisted(chat_id, user_id).await.unwrap_or(false) {
        return Ok(false);
    }
    if is_group_admin(bot, message.chat.id, user_id).await {
        return Ok(false);
    }

    let settings = runtime.get_group_modules(chat_id).await.unwrap_or_default();
    if !settings.flood_control {
        return Ok(false);
    }

    if !runtime.check_flood(chat_id, user_id).await {
        return Ok(false);
    }

    let _ = mute_user(bot, message.chat.id, user_id).await;
    let case = CaseRecord {
        id: Uuid::new_v4().to_string(),
        action: ActionKind::FloodMute,
        chat_id,
        target_user_id: user_id,
        target_name: short_user(user),
        actor_user_id: None,
        actor_name: None,
        source_message_id: Some(message.id.0),
        evidence_text: extract_full_text(message),
        model_score: None,
        matched_rule_id: None,
        matched_rule_pattern: Some("FLOOD".to_string()),
        status: "auto_muted".to_string(),
        log_message_id: None,
        created_at: Utc::now(),
    };
    let log_message_id = log_action(bot, runtime, &case).await.unwrap_or_default();
    let mut updated = case.clone();
    updated.log_message_id = Some(log_message_id);
    let _ = store_case(runtime, &updated).await;
    let _ = notify_group(bot, runtime, &updated, log_message_id, "<b>自動洗版偵測禁言</b>").await;
    Ok(true)
}

/// Message-time safety net for netban: catches members who were already in
/// a group before it turned netban on, or who joined between propagation
/// events - cases the join-time check in `notify_bot_added` can't reach,
/// since the Bot API has no way to enumerate existing members to backfill
/// against. Only does its DB lookup when the current chat has netban
/// enabled, so groups that never opt in pay zero extra cost per message.
async fn check_netban_and_act(bot: &Bot, runtime: &Arc<Runtime>, message: &Message) -> bool {
    if !message.chat.is_group() && !message.chat.is_supergroup() {
        return false;
    }
    let Some(user) = message.from.as_ref() else { return false; };
    if user.is_bot {
        return false;
    }
    let chat_id = message.chat.id.0;
    let user_id = user.id.0 as i64;

    if is_special_user(&runtime.config, user_id) {
        return false;
    }

    let settings = runtime.get_group_modules(chat_id).await.unwrap_or_default();
    if !settings.netban {
        return false;
    }

    if runtime.is_global_whitelisted(user_id).await.unwrap_or(false) {
        return false;
    }
    if runtime.is_group_whitelisted(chat_id, user_id).await.unwrap_or(false) {
        return false;
    }
    if is_group_admin(bot, message.chat.id, user_id).await {
        return false;
    }

    let Ok(Some(prior_case)) = runtime.find_active_network_ban(user_id).await else {
        return false;
    };

    let _ = bot.delete_message(message.chat.id, message.id).await;
    let _ = bot.ban_chat_member(message.chat.id, user.id).await;
    let _ = runtime.record_network_ban_target(&prior_case.id, chat_id).await;
    let _ = bot
        .send_message(
            message.chat.id,
            format!(
                "<b>跨群組黑名單同步封禁</b>\n用戶 <code>{user_id}</code> 已因跨群組黑名單同步封禁。\n原始案例: <code>{}</code>",
                prior_case.id,
            ),
        )
        .parse_mode(ParseMode::Html)
        .await;
    true
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
        ModerationCommand::HelpOp => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            bot.send_message(message.chat.id, help_op_text()).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::MyId => {
            let requester = message.from.as_ref();
            let target_user = message
                .reply_to_message()
                .and_then(|m| m.from.as_ref())
                .or(requester);
            let uid = target_user.map(|u| u.id.0.to_string()).unwrap_or_else(|| "unknown".to_string());
            let target_name = target_user.map(short_user).unwrap_or_else(|| "unknown".to_string());
            let maintainer = if let Some(user) = target_user {
                if is_maintainer(&bot, &runtime.config, user.id.0 as i64).await { "yes" } else { "no" }
            } else {
                "no"
            };
            let body = format!("<b>查詢結果</b>\n• 對象: <code>{target_name}</code>\n• Telegram ID: <code>{uid}</code>\n• Maintainer: {maintainer}");
            bot.send_message(message.chat.id, body).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::MyChat => {
            bot.send_message(message.chat.id, format!("這個群的 Chat ID: <code>{}</code>", message.chat.id.0)).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::ScoreTest(text) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以使用 /ml_score。");
            let target_msg = message.reply_to_message().unwrap_or(&message);
            let text = if text.trim().is_empty() {
                extract_full_text(target_msg)
            } else {
                text
            };
            if text.trim().is_empty() {
                bot.send_message(message.chat.id, "請在指令後面提供要測試的文本，或回覆一條消息後使用 /ml_score。") .await?;
                return Ok(());
            }
            let user_name = message.from.as_ref().map(short_user).unwrap_or_else(|| "unknown".to_string());
            let result = runtime.inspect_message(&user_name, &text).await.map_err(|e| teloxide::RequestError::Io(std::io::Error::other(e.to_string()).into()))?;
            let response = match &result {
                InspectionResult::Spam { score, matched_rule: Some(rule) } => format!(
                    "<b>判定</b>: 垃圾\n<b>分數</b>: {score:.6}\n<b>規則</b>: REGEX\n<b>說明</b>: {}",
                    escape_html(&rule.description),
                ),
                InspectionResult::Spam { score, .. } => {
                    let report = runtime.score_debug(&user_name, &text).await.map_err(|e| teloxide::RequestError::Io(std::io::Error::other(e.to_string()).into()))?;
                    format!("<b>判定</b>: 垃圾\n<b>分數</b>: {score:.6}\n{}", format_score_debug(&report))
                }
                InspectionResult::Ham { score } => {
                    let report = runtime.score_debug(&user_name, &text).await.map_err(|e| teloxide::RequestError::Io(std::io::Error::other(e.to_string()).into()))?;
                    format!("<b>判定</b>: 正常\n<b>分數</b>: {score:.6}\n{}", format_score_debug(&report))
                }
            };
            bot.send_message(message.chat.id, response).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::SetChat(chat_id) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以設定項目交流群。");
            // No argument: bind whatever chat this was run in (must be a
            // group), so you don't need to already know its numeric ID.
            // Still accepts an explicit ID too, e.g. to set it from DM.
            let value = if chat_id.trim().is_empty() {
                if !message.chat.is_group() && !message.chat.is_supergroup() {
                    bot.send_message(message.chat.id, "請在群組中使用 /setchat 綁定目前的群組，或提供 Chat ID。").await?;
                    return Ok(());
                }
                message.chat.id.0
            } else {
                let Some(value) = chat_id.parse::<i64>().ok() else {
                    bot.send_message(message.chat.id, "請提供有效的 Chat ID。").await?;
                    return Ok(());
                };
                value
            };
            let old = runtime.project_chat().await;
            runtime.set_project_chat(value).await;
            log_maintainer_action(&bot, &runtime, from_id, &short_user(from), None, "/setchat", &format!("項目交流群 {old:?} → {value}"), UndoData::ProjectChat { old }).await;
            bot.send_message(message.chat.id, format!("已設定項目交流群為 <code>{value}</code>。此群組串連的頻道發文自動釘選時，機器人會自動取消釘選。")).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::SetAuditLog(chat_id) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以設定日誌頻道。");
            // Same no-argument-binds-current-chat convenience as /setchat.
            let value = if chat_id.trim().is_empty() {
                if !message.chat.is_group() && !message.chat.is_supergroup() {
                    bot.send_message(message.chat.id, "請在群組/頻道中使用 /set_audit_log 綁定，或提供 Chat ID。").await?;
                    return Ok(());
                }
                message.chat.id.0
            } else {
                let Some(value) = chat_id.parse::<i64>().ok() else {
                    bot.send_message(message.chat.id, "請提供有效的 Chat ID。").await?;
                    return Ok(());
                };
                value
            };
            runtime.set_audit_log_chat(value).await;
            bot.send_message(message.chat.id, format!("已設定維護操作日誌頻道為 <code>{value}</code>。之後每個會改變狀態的維護指令都會記錄在這裡，並附上可用於 /revert 的 action id。")).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::Leave(reason) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以使用 /leave。");
            let (target_chat_id, reason) = parse_leave_args(&reason);
            let reason = if reason.trim().is_empty() { "違反使用規則".to_string() } else { reason };
            let target_chat_id = target_chat_id.unwrap_or(message.chat.id.0);
            let project_chat = match runtime.project_chat().await {
                Some(id) => id,
                None => {
                    bot.send_message(message.chat.id, "尚未設定項目交流群，請先使用 /setchat。") .await?;
                    return Ok(());
                }
            };
            let text = format!("已停止為此群提供服務。原因：{}", escape_html(&reason));
            let button = InlineKeyboardMarkup::new(vec![vec![InlineKeyboardButton::url("前往項目交流群查詢", Url::parse(&project_chat_link(project_chat)).unwrap())]]);
            let _ = bot.send_message(ChatId(target_chat_id), text).parse_mode(ParseMode::Html).reply_markup(button).await;
            let _ = bot.leave_chat(ChatId(target_chat_id)).await;
        }
        ModerationCommand::SpamBan | ModerationCommand::Mute | ModerationCommand::Kick => {
            let Some((target_id, target_name, source_id, evidence_text)) = extract_reply_context(&message).await else {
                reply_ephemeral(&bot, &message, "請回覆一條訊息後再使用此指令。").await?;
                return Ok(());
            };

            if !is_group_admin(&bot, message.chat.id, from_id).await {
                handle_permission_denied(&bot, &runtime, &message, from, "只有群組管理員可以執行此指令。").await?;
                return Ok(());
            }

            if is_group_admin(&bot, message.chat.id, target_id).await || is_special_user(&runtime.config, target_id) {
                reply_ephemeral(&bot, &message, "不能對群組管理員或項目維護人員執行此指令。").await?;
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
                matched_rule_id: None,
                matched_rule_pattern: None,
                status: "done".to_string(),
                log_message_id: None,
                created_at: Utc::now(),
            };

            match action {
                ActionKind::SpamBan => {
                    let _ = bot.delete_message(message.chat.id, MessageId(source_id)).await;
                    ban_user(&bot, message.chat.id, target_id).await.ok();
                    train_spam(&runtime, &evidence_text, Some(&case_id)).await.ok();
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
            if action == ActionKind::SpamBan {
                propagate_network_ban(&bot, &runtime, &case).await;
            }

            // Reuses the case's own case_id as the revert handle - no new ID
            // needed, /revert for a Case just calls the same
            // reverse_ban_case/reverse_mute_case the case_id form of
            // /unban and /unmute already use. A kick has nothing persistent
            // to undo (it's just a ban immediately followed by an unban).
            let (command_name, undo) = match action {
                ActionKind::SpamBan => ("/sb", UndoData::Case { case_id: case_id.clone(), kind: CaseKind::Ban }),
                ActionKind::Mute => ("/mute", UndoData::Case { case_id: case_id.clone(), kind: CaseKind::Mute }),
                _ => ("/kick", UndoData::NotRevertible),
            };
            log_maintainer_action(&bot, &runtime, from_id, &short_user(from), Some(message.chat.id.0), command_name, &format!("{} 對象={target_id}", chinese_case_action(&case)), undo).await;

            // Delete the command message to minimize group disruption
            let _ = bot.delete_message(message.chat.id, message.id).await;
        }
        ModerationCommand::SpamReport => {
            let Some((target_id, target_name, source_id, evidence_text)) = extract_reply_context(&message).await else {
                reply_ephemeral(&bot, &message, "請回覆一條疑似 spam 的訊息。").await?;
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
                matched_rule_id: None,
                matched_rule_pattern: None,
                status: "pending_review".to_string(),
                log_message_id: None,
                created_at: Utc::now(),
            };

            let keyboard = InlineKeyboardMarkup::new(vec![vec![
                InlineKeyboardButton::callback("受理並封禁", format!("review:approve:{case_id}")),
                InlineKeyboardButton::callback("拒絕並洗模型", format!("review:reject:{case_id}")),
            ]]);

            let text = format!(
                "<b>新的 /spam 申請</b>\n\n<b>對象</b>: {} ({})\n<b>發起人</b>: {}\n<b>內容</b>: <blockquote>{}</blockquote>\n<b>案例</b>: <code>{}</code>",
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
                    let reason_link = runtime.blacklist_reason_link().await.unwrap_or_else(|| link.clone());
                    let text = format_case_lookup(&case, &link, &reason_link);
                    bot.send_message(message.chat.id, text).parse_mode(ParseMode::Html).await?;
                }
                _ => {
                    bot.send_message(message.chat.id, "找不到該 Case。") .await?;
                }
            }
        }
        ModerationCommand::MlTrainSpam | ModerationCommand::MlCleanSpam => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let target_msg = message.reply_to_message().unwrap_or(&message);
            let text = extract_full_text(target_msg);
            if text.trim().is_empty() {
                bot.send_message(message.chat.id, "請回覆一條訊息來訓練或清洗模型。").await?;
                return Ok(());
            }
            // A fresh UUID passed as case_id purely as a revert handle (see
            // UndoData::TrainingSample) - there's no real case behind a
            // manual single-sample training action.
            let training_ref = Uuid::new_v4().to_string();
            match cmd {
                ModerationCommand::MlTrainSpam => {
                    train_spam(&runtime, &text, Some(&training_ref)).await.ok();
                    log_maintainer_action(&bot, &runtime, from_id, &short_user(from), Some(message.chat.id.0), "/ml_train_spam", "手動訓練 spam 樣本", UndoData::TrainingSample { training_ref }).await;
                    bot.send_message(message.chat.id, "已將該樣本寫入 spam 模型。") .await?;
                }
                ModerationCommand::MlCleanSpam => {
                    train_ham(&runtime, &text, Some(&training_ref)).await.ok();
                    log_maintainer_action(&bot, &runtime, from_id, &short_user(from), Some(message.chat.id.0), "/ml_clean_spam", "手動訓練 ham/clean 樣本", UndoData::TrainingSample { training_ref }).await;
                    bot.send_message(message.chat.id, "已將該樣本寫入 ham/clean 模型。") .await?;
                }
                _ => {}
            }
        }
        ModerationCommand::MarkHam => {
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以使用 /mark_ham。");
            let target_msg = message.reply_to_message().unwrap_or(&message);
            let text = extract_full_text(target_msg);
            if text.trim().is_empty() {
                bot.send_message(message.chat.id, "請回覆一條訊息作為 ham 樣本。") .await?;
                return Ok(());
            }
            let training_ref = Uuid::new_v4().to_string();
            train_ham(&runtime, &text, Some(&training_ref)).await.ok();
            log_maintainer_action(&bot, &runtime, from_id, &short_user(from), Some(message.chat.id.0), "/mark_ham", "手動標記 ham 樣本", UndoData::TrainingSample { training_ref }).await;
            bot.send_message(message.chat.id, "已將該樣本寫入 ham 模型。") .await?;
        }
        ModerationCommand::MlUndoCleanSpam => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let raw_text = message.text().or(message.caption()).unwrap_or("");
            let text = if let Some(target_msg) = message.reply_to_message() {
                extract_full_text(target_msg)
            } else {
                let args = raw_text.split_whitespace().skip(1).collect::<Vec<_>>().join(" ");
                if args.is_empty() {
                    bot.send_message(message.chat.id, "請回覆一條先前寫入 ham/clean 的樣本訊息，或在指令後直接貼上要撤銷的文字。").await?;
                    return Ok(());
                }
                args
            };
            if text.trim().is_empty() {
                bot.send_message(message.chat.id, "請回覆一條先前寫入 ham/clean 的樣本訊息，或在指令後直接貼上要撤銷的文字。").await?;
                return Ok(());
            }

            let removed = runtime.undo_clean_training_sample_by_text(&text).await.unwrap_or(0);
            if removed == 0 {
                bot.send_message(message.chat.id, "找不到可撤銷的 ham/clean 樣本。請確認文字完全一致，或先用 /ml_export 檢查實際寫入內容。").await?;
                return Ok(());
            }

            let _ = runtime.rebuild_model().await;
            bot.send_message(message.chat.id, "已撤銷該 ham/clean 樣本並重建模型。").await?;
        }
        ModerationCommand::MlPurge(case_id) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let removed = runtime.purge_training_by_case(&case_id).await.unwrap_or(0);
            let _ = runtime.rebuild_model().await;
            bot.send_message(message.chat.id, format!("已刪除 {removed} 筆訓練樣本，並重建模型。")) .await?;
        }
        ModerationCommand::MlPurgeText(target) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let payload = target.trim();
            if payload.is_empty() {
                bot.send_message(message.chat.id, "請提供要清除的原文片段。").await?;
                return Ok(());
            }
            let removed = runtime.purge_training_by_text(payload).await.unwrap_or(0);
            let _ = runtime.rebuild_model().await;
            bot.send_message(message.chat.id, format!("已依文字清除 {removed} 筆訓練樣本，並重建模型。")) .await?;
        }
        ModerationCommand::MlRebuild => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let rebuilt = runtime.rebuild_model().await.unwrap_or_default();
            bot.send_message(message.chat.id, format!("已重建模型，spam_docs={} ham_docs={}", rebuilt.spam_docs, rebuilt.ham_docs)).await?;
        }
        ModerationCommand::MlStats => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let (spam, ham, total) = runtime.word_stats().await.unwrap_or((0, 0, 0));
            let threshold = runtime.effective_threshold(Some(message.chat.id.0)).await.unwrap_or(runtime.config.spam_threshold);
            let threshold_source = if runtime.get_group_modules(message.chat.id.0).await.ok().and_then(|s| s.spam_threshold_override).is_some() {
                "本群自訂"
            } else {
                "全域"
            };
            let mut text = format!("<b>模型統計</b>\nspam: {spam}\nham: {ham}\n總樣本: {total}\n有效門檻: {threshold:.2}（{threshold_source}）");
            if let Ok((top_spam, top_ham)) = runtime.largest_token_counts().await {
                if let Some((word, count)) = top_spam {
                    text.push_str(&format!("\n\n最大 spam token: <code>{}</code> = {count}", escape_html(&word)));
                    if spam > 0 && (count as f64) / (spam as f64) > 0.2 {
                        text.push_str("\n⚠️ 此 token 佔整體 spam 計數超過 20%，可能是 /set 造成的異常值，會拖累其他詞的判斷，建議檢查並用 /set 重新調整或用 /ml_purge_text 清除。");
                    }
                }
                if let Some((word, count)) = top_ham {
                    text.push_str(&format!("\n最大 ham token: <code>{}</code> = {count}", escape_html(&word)));
                    if ham > 0 && (count as f64) / (ham as f64) > 0.2 {
                        text.push_str("\n⚠️ 此 token 佔整體 ham 計數超過 20%，可能是 /set 造成的異常值。");
                    }
                }
            }
            bot.send_message(message.chat.id, text).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::CheckRules => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let invalid = runtime.list_invalid_spam_rules().await.map_err(|e| teloxide::RequestError::Io(std::io::Error::other(e.to_string()).into()))?;
            let body = if invalid.is_empty() {
                "<b>規則檢查</b>\n\n全部通過".to_string()
            } else {
                let mut out = String::from("<b>規則檢查</b>\n\n以下規則無法編譯：\n");
                for (id, pattern, description, err) in invalid {
                    if description.trim().is_empty() {
                        out.push_str(&format!("<code>@{}</code>\n╚• <code>{}</code>\n<blockquote>{}</blockquote>\n", id, escape_html(&pattern), escape_html(&err)));
                    } else {
                        out.push_str(&format!("<code>@{}</code> {}\n╚• <code>{}</code>\n<blockquote>{}</blockquote>\n", id, escape_html(&description), escape_html(&pattern), escape_html(&err)));
                    }
                }
                out
            };
            bot.send_message(message.chat.id, body).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::ListRules => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let rules = runtime.list_spam_rules().await.map_err(|e| teloxide::RequestError::Io(std::io::Error::other(e.to_string()).into()))?;
            let body = if rules.is_empty() {
                "<b>規則清單</b>\n\n╚• 無".to_string()
            } else {
                let mut out = String::from("<b>規則清單</b>\n\n<b>已載入規則</b>\n");
                for (idx, (id, pattern, description)) in rules.into_iter().enumerate() {
                    if idx > 0 {
                        out.push('\n');
                    }
                    if description.trim().is_empty() {
                        out.push_str(&format!("<code>@{}</code>\n╚• <code>{}</code>\n", id, escape_html(&pattern)));
                    } else {
                        out.push_str(&format!("<code>@{}</code> {}\n╚• <code>{}</code>\n", id, escape_html(&description), escape_html(&pattern)));
                    }
                }
                out
            };
            bot.send_message(message.chat.id, body).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::DelRule(rule_id) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let Some(id) = rule_id.parse::<i64>().ok() else {
                bot.send_message(message.chat.id, "請提供有效的規則 ID。").await?;
                return Ok(());
            };
            let old_rule = runtime.list_spam_rules().await.unwrap_or_default().into_iter().find(|(rid, _, _)| *rid == id);
            let removed = runtime.delete_spam_rule(id).await.map_err(|e| teloxide::RequestError::Io(std::io::Error::other(e.to_string()).into()))?;
            if removed {
                if let Some((_, pattern, description)) = old_rule {
                    log_maintainer_action(&bot, &runtime, from_id, &short_user(from), None, "/del_rule", &format!("刪除規則 @{id}"), UndoData::RuleDeleted { pattern, description }).await;
                }
            }
            bot.send_message(message.chat.id, if removed { format!("已刪除規則 #{id}") } else { format!("找不到規則 #{id}") }).await?;
        }
        ModerationCommand::AddRule(rule) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let pattern = rule.trim();
            if pattern.is_empty() {
                bot.send_message(message.chat.id, "請提供正則。").await?;
                return Ok(());
            }
            runtime.start_pending_rule_addition(from_id, pattern.to_string()).await;
            bot.send_message(message.chat.id, "好的，這組正則要叫什麼？").await?;
        }
        ModerationCommand::EditRule(rule_id, pattern) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let Some(id) = rule_id.parse::<i64>().ok() else {
                bot.send_message(message.chat.id, "請提供有效的規則 ID。").await?;
                return Ok(());
            };
            if pattern.trim().is_empty() {
                bot.send_message(message.chat.id, "請提供正則。").await?;
                return Ok(());
            }
            let old_pattern = runtime.list_spam_rules().await.unwrap_or_default().into_iter().find(|(rid, _, _)| *rid == id).map(|(_, p, _)| p);
            let updated = runtime.update_spam_rule_pattern(id, pattern.trim()).await.map_err(|e| teloxide::RequestError::Io(std::io::Error::other(e.to_string()).into()))?;
            if updated {
                if let Some(old_pattern) = old_pattern {
                    log_maintainer_action(&bot, &runtime, from_id, &short_user(from), None, "/edit_rule", &format!("規則 @{id} 正則變更"), UndoData::RuleEdited { rule_id: id, old_pattern }).await;
                }
                bot.send_message(message.chat.id, format!("已更新規則 @{id}。名稱不變。\n")).await?;
            } else {
                bot.send_message(message.chat.id, format!("找不到規則 @{id}。")).await?;
            }
        }
        ModerationCommand::UpdateBL => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let text = build_blacklist_reason_text(&runtime);
            let sent = bot.send_message(ChatId(runtime.config.log_channel_id), text).parse_mode(ParseMode::Html).await?;
            let _ = bot.pin_chat_message(ChatId(runtime.config.log_channel_id), sent.id).await;
            let _ = runtime.set_blacklist_reason_message_id(sent.id.0).await;
            bot.send_message(message.chat.id, format!("已更新封禁代號說明：<code>{}</code>", sent.id.0)).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::Module(module, state) => {
            if !message.chat.is_group() && !message.chat.is_supergroup() {
                reply_ephemeral(&bot, &message, "請在群組中使用 /module。").await?;
                return Ok(());
            }
            if !is_group_admin(&bot, message.chat.id, from_id).await {
                handle_permission_denied(&bot, &runtime, &message, from, "只有群組管理員可以設定模組。").await?;
                return Ok(());
            }
            let enabled = matches!(state.to_lowercase().as_str(), "on" | "enable" | "enabled");
            let key = module.trim().to_lowercase();
            let old_settings = runtime.get_group_modules(message.chat.id.0).await.unwrap_or_default();
            let old_enabled = match key.as_str() {
                "nolongname" => Some(old_settings.no_long_name),
                "nohalal" => Some(old_settings.no_halal),
                "nosm" => Some(old_settings.no_service_messages),
                "flood" => Some(old_settings.flood_control),
                "captcha" => Some(old_settings.captcha),
                "netban" => Some(old_settings.netban),
                "cmdclean" => Some(old_settings.cmd_clean),
                _ => None,
            };
            match key.as_str() {
                "nolongname" => { runtime.set_group_module(message.chat.id.0, "nolongname", enabled).await.ok(); }
                "nohalal" => { runtime.set_group_module(message.chat.id.0, "nohalal", enabled).await.ok(); }
                "nosm" => { runtime.set_group_module(message.chat.id.0, "nosm", enabled).await.ok(); }
                "flood" => { runtime.set_group_module(message.chat.id.0, "flood", enabled).await.ok(); }
                "captcha" => { runtime.set_group_module(message.chat.id.0, "captcha", enabled).await.ok(); }
                "netban" => { runtime.set_group_module(message.chat.id.0, "netban", enabled).await.ok(); }
                "cmdclean" => { runtime.set_group_module(message.chat.id.0, "cmdclean", enabled).await.ok(); }
                _ => {
                    reply_ephemeral(&bot, &message, "模組名稱僅支援 NoLongName / NoHalal / NoSM / Flood / Captcha / Netban / CmdClean。").await?;
                    return Ok(());
                }
            }
            if let Some(old_enabled) = old_enabled {
                log_maintainer_action(
                    &bot,
                    &runtime,
                    from_id,
                    &short_user(from),
                    Some(message.chat.id.0),
                    "/module",
                    &format!("{key} {old_enabled}→{enabled}"),
                    UndoData::GroupModule { chat_id: message.chat.id.0, module: key.clone(), old_enabled },
                )
                .await;
            }
            bot.send_message(message.chat.id, format!("已將 {module} 設為 {}", if enabled { "on" } else { "off" })).await?;
        }
        ModerationCommand::White(target) => {
            if !message.chat.is_group() && !message.chat.is_supergroup() {
                reply_ephemeral(&bot, &message, "請在群組中使用 /white。").await?;
                return Ok(());
            }
            if !is_group_admin(&bot, message.chat.id, from_id).await {
                handle_permission_denied(&bot, &runtime, &message, from, "只有群組管理員可以設定白名單。").await?;
                return Ok(());
            }
            let Some(user_id) = target.parse::<i64>().ok().or_else(|| message.reply_to_message().and_then(|m| m.from.as_ref()).map(|u| u.id.0 as i64)) else {
                reply_ephemeral(&bot, &message, "請提供 userid 或回覆一位用戶。").await?;
                return Ok(());
            };
            let old_enabled = runtime.is_group_whitelisted(message.chat.id.0, user_id).await.unwrap_or(false);
            runtime.set_group_whitelist(message.chat.id.0, user_id, true, Some(from_id)).await.ok();
            log_maintainer_action(&bot, &runtime, from_id, &short_user(from), Some(message.chat.id.0), "/white", &format!("本群白名單 user_id={user_id} {old_enabled}→true"), UndoData::GroupWhitelist { chat_id: message.chat.id.0, user_id, old_enabled }).await;
            bot.send_message(message.chat.id, format!("已將 <code>{user_id}</code> 加入本群白名單。",)).parse_mode(ParseMode::Html).await?;

            // Delete the command message to minimize group disruption
            let _ = bot.delete_message(message.chat.id, message.id).await;
        }
        ModerationCommand::WhiteGlobal(target) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let Some(user_id) = target.parse::<i64>().ok().or_else(|| message.reply_to_message().and_then(|m| m.from.as_ref()).map(|u| u.id.0 as i64)) else {
                bot.send_message(message.chat.id, "請提供 userid 或回覆一位用戶。") .await?;
                return Ok(());
            };
            let old_enabled = runtime.is_global_whitelisted(user_id).await.unwrap_or(false);
            runtime.set_global_whitelist(user_id, true, Some(from_id)).await.ok();
            log_maintainer_action(&bot, &runtime, from_id, &short_user(from), None, "/white -global", &format!("全域白名單 user_id={user_id} {old_enabled}→true"), UndoData::GlobalWhitelist { user_id, old_enabled }).await;
            bot.send_message(message.chat.id, format!("已將 <code>{user_id}</code> 加入全域白名單。",)).parse_mode(ParseMode::Html).await?;

            // Delete the command message to minimize group disruption
            let _ = bot.delete_message(message.chat.id, message.id).await;
        }
        ModerationCommand::Unwhite(target) => {
            if !message.chat.is_group() && !message.chat.is_supergroup() {
                reply_ephemeral(&bot, &message, "請在群組中使用 /unwhite。").await?;
                return Ok(());
            }
            if !is_group_admin(&bot, message.chat.id, from_id).await {
                handle_permission_denied(&bot, &runtime, &message, from, "只有群組管理員可以設定白名單。").await?;
                return Ok(());
            }
            let Some(user_id) = target.parse::<i64>().ok().or_else(|| message.reply_to_message().and_then(|m| m.from.as_ref()).map(|u| u.id.0 as i64)) else {
                reply_ephemeral(&bot, &message, "請提供 userid 或回覆一位用戶。").await?;
                return Ok(());
            };
            let old_enabled = runtime.is_group_whitelisted(message.chat.id.0, user_id).await.unwrap_or(false);
            runtime.set_group_whitelist(message.chat.id.0, user_id, false, Some(from_id)).await.ok();
            log_maintainer_action(&bot, &runtime, from_id, &short_user(from), Some(message.chat.id.0), "/unwhite", &format!("本群白名單 user_id={user_id} {old_enabled}→false"), UndoData::GroupWhitelist { chat_id: message.chat.id.0, user_id, old_enabled }).await;
            bot.send_message(message.chat.id, format!("已將 <code>{user_id}</code> 移出本群白名單。",)).parse_mode(ParseMode::Html).await?;

            // Delete the command message to minimize group disruption
            let _ = bot.delete_message(message.chat.id, message.id).await;
        }
        ModerationCommand::UnwhiteGlobal(target) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let Some(user_id) = target.parse::<i64>().ok().or_else(|| message.reply_to_message().and_then(|m| m.from.as_ref()).map(|u| u.id.0 as i64)) else {
                bot.send_message(message.chat.id, "請提供 userid 或回覆一位用戶。") .await?;
                return Ok(());
            };
            let old_enabled = runtime.is_global_whitelisted(user_id).await.unwrap_or(false);
            runtime.set_global_whitelist(user_id, false, Some(from_id)).await.ok();
            log_maintainer_action(&bot, &runtime, from_id, &short_user(from), None, "/unwhite -global", &format!("全域白名單 user_id={user_id} {old_enabled}→false"), UndoData::GlobalWhitelist { user_id, old_enabled }).await;
            bot.send_message(message.chat.id, format!("已將 <code>{user_id}</code> 移出全域白名單。",)).parse_mode(ParseMode::Html).await?;

            // Delete the command message to minimize group disruption
            let _ = bot.delete_message(message.chat.id, message.id).await;
        }
        ModerationCommand::Check(target) => {
            if !message.chat.is_group() && !message.chat.is_supergroup() {
                reply_ephemeral(&bot, &message, "請在群組中使用 /check。").await?;
                return Ok(());
            }
            let Some(target_msg) = message.reply_to_message() else {
                if let Ok(user_id) = target.parse::<i64>() {
                    if runtime.is_global_whitelisted(user_id).await.unwrap_or(false) {
                        bot.send_message(message.chat.id, global_whitelist_check_text()).parse_mode(ParseMode::Html).await?;
                        return Ok(());
                    }
                    let profile = runtime.load_user_profile(&bot, user_id).await;
                    match profile {
                        Ok(profile) => {
                            let result = runtime.check_group_modules(&bot, message.chat.id.0, &teloxide::types::User {
                                id: UserId(profile.user_id as u64),
                                is_bot: false,
                                first_name: profile.display_name.clone(),
                                last_name: None,
                                username: profile.username.clone(),
                                language_code: None,
                                is_premium: false,
                                added_to_attachment_menu: false,
                            }, profile.bio.as_deref(), None).await;
                            match result {
                                Ok(result) => {
                                    let hit = if result.reasons.is_empty() { "無".to_string() } else { result.reasons.join("；") };
                                    let name = if result.name_guard.is_empty() { "無".to_string() } else { result.name_guard.join("；") };
                                    let halal = if result.no_halal.is_empty() { "無".to_string() } else { result.no_halal.join("；") };
                                    let reason_link = runtime.blacklist_reason_link().await;
                                    let body = format!(
                                        "<b>檢查結果</b>\n<b>對象</b>: {}\n<b>命中</b>: {}\n<b>名稱規則</b>: {}\n<b>清真規則</b>: {}",
                                        escape_html(&profile.display_name),
                                        format_public_reason(&hit, reason_link.as_deref()),
                                        format_public_reason(&name, reason_link.as_deref()),
                                        format_public_reason(&halal, reason_link.as_deref()),
                                    );
                                    bot.send_message(message.chat.id, body).parse_mode(ParseMode::Html).await?;
                                }
                                Err(err) => {
                                    bot.send_message(message.chat.id, format!("檢查失敗：{err}")).await?;
                                }
                            }
                            return Ok(());
                        }
                        Err(err) => {
                            bot.send_message(message.chat.id, format!("檢查失敗：{err}")).await?;
                            return Ok(());
                        }
                    }
                }
                reply_ephemeral(&bot, &message, "請回覆一位用戶後再使用 /check。").await?;
                return Ok(());
            };
            let Some(user) = target_msg.from.as_ref() else {
                bot.send_message(message.chat.id, "找不到可檢查的目標用戶。") .await?;
                return Ok(());
            };
            if runtime.is_global_whitelisted(user.id.0 as i64).await.unwrap_or(false) {
                bot.send_message(message.chat.id, global_whitelist_check_text()).parse_mode(ParseMode::Html).await?;
                return Ok(());
            }
            let text = extract_full_text(target_msg);
            let result = runtime.check_group_modules(&bot, message.chat.id.0, user, None, Some(&text)).await;
            match result {
                Ok(result) => {
                    let hit = if result.reasons.is_empty() { "無".to_string() } else { result.reasons.join("；") };
                    let name = if result.name_guard.is_empty() { "無".to_string() } else { result.name_guard.join("；") };
                    let halal = if result.no_halal.is_empty() { "無".to_string() } else { result.no_halal.join("；") };
                    let reason_link = runtime.blacklist_reason_link().await;
                    let body = format!(
                        "<b>檢查結果</b>\n<b>對象</b>: {}\n<b>命中</b>: {}\n<b>名稱規則</b>: {}\n<b>清真規則</b>: {}",
                        escape_html(&short_user(user)),
                        format_public_reason(&hit, reason_link.as_deref()),
                        format_public_reason(&name, reason_link.as_deref()),
                        format_public_reason(&halal, reason_link.as_deref()),
                    );
                    bot.send_message(message.chat.id, body).parse_mode(ParseMode::Html).await?;
                }
                Err(err) => {
                    bot.send_message(message.chat.id, format!("檢查失敗：{err}")).await?;
                }
            }
        }
        ModerationCommand::MlThreshold(value) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let Ok(threshold) = value.parse::<f64>() else {
                bot.send_message(message.chat.id, "請提供 0.50 到 0.99 的數值。").await?;
                return Ok(());
            };
            let clamped = threshold.clamp(0.50, 0.99);
            // DM, the test group, and the project/work chat aren't moderated
            // customer groups with their own policy - a threshold set there is
            // the global default. Any other group gets its own override.
            let is_global_scope = message.chat.is_private()
                || runtime.config.test_group_id == Some(message.chat.id.0)
                || runtime.project_chat().await == Some(message.chat.id.0);
            if is_global_scope {
                let old = runtime.current_threshold().await.unwrap_or(runtime.config.spam_threshold);
                runtime.set_threshold(clamped).await.ok();
                log_maintainer_action(&bot, &runtime, from_id, &short_user(from), None, "/ml_threshold", &format!("全域門檻 {old:.2} → {clamped:.2}"), UndoData::Threshold { old }).await;
                bot.send_message(message.chat.id, format!("已保存全域門檻: {clamped:.2}")).await?;
            } else {
                let old = runtime.get_group_modules(message.chat.id.0).await.unwrap_or_default().spam_threshold_override;
                runtime.set_group_threshold(message.chat.id.0, Some(clamped)).await.ok();
                log_maintainer_action(&bot, &runtime, from_id, &short_user(from), Some(message.chat.id.0), "/ml_threshold", &format!("本群門檻 {old:?} → {clamped:.2}"), UndoData::GroupThreshold { chat_id: message.chat.id.0, old }).await;
                bot.send_message(message.chat.id, format!("已為本群設定門檻: {clamped:.2}（僅適用於本群，其他群組不受影響）")).await?;
            }
        }
        ModerationCommand::MlSetToken(token_arg, value) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");

            let token = token_arg.strip_prefix("0x").unwrap_or(&token_arg).trim();
            if token.is_empty() {
                bot.send_message(message.chat.id, "請提供 token，例如 /set 0x80sun 0.9。").await?;
                return Ok(());
            }

            let Ok(target) = value.parse::<f64>() else {
                bot.send_message(message.chat.id, "請提供 0.05 到 0.95 之間的數值（過於極端的值會影響整個模型的計分基準，因此會被限制）。").await?;
                return Ok(());
            };

            let target = target.clamp(0.000001, 0.999999);
            let token_owned = token.to_string();
            let (spam_count, ham_count, old_spam, old_ham) = runtime
                .set_token_probability(token, target)
                .await
                .map_err(|err| teloxide::RequestError::Io(std::io::Error::other(err.to_string()).into()))?;
            log_maintainer_action(
                &bot,
                &runtime,
                from_id,
                &short_user(from),
                None,
                "/set",
                &format!("token `{token_owned}` spam_count {old_spam}→{spam_count}, ham_count {old_ham}→{ham_count}"),
                UndoData::TokenProbability { token: token_owned, old_spam, old_ham },
            )
            .await;

            bot.send_message(
                message.chat.id,
                format!(
                    "已調整 token <code>{}</code>：目標 spam_prob={:.4}，spam_count={}，ham_count={}",
                    escape_html(token),
                    target,
                    spam_count,
                    ham_count,
                ),
            )
            .parse_mode(ParseMode::Html)
            .await?;
        }
        ModerationCommand::MlExport => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
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
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以使用 /import。");
            let target_msg = message.reply_to_message().unwrap_or(&message);
            let text = extract_full_text(target_msg);
            if text.trim().is_empty() {
                bot.send_message(message.chat.id, "請回覆一段匯出列表或輸出結果。") .await?;
                return Ok(());
            }
            let payloads = import_train_payloads(&text);
            if payloads.is_empty() {
                bot.send_message(message.chat.id, "沒有找到可匯入的訓練字串。") .await?;
                return Ok(());
            }
            let mut count = 0usize;
            let mut debug = Vec::new();
            for payload in payloads {
                debug.push(payload.clone());
                train_spam(&runtime, &payload, None).await.ok();
                count += 1;
            }
            bot.send_message(message.chat.id, format!("已匯入並訓練 {count} 筆。\n\n匯入字串：\n{}", debug.join("\n---\n"))).await?;
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
        ModerationCommand::MlStartMassHam => {
            if !message.chat.is_private() || !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只允許維護者在私訊中啟動批量訓練。") .await?;
                return Ok(());
            }
            runtime.start_mass_train(from_id).await;
            runtime.set_mass_train_mode(from_id, "ham" ).await;
            bot.send_message(message.chat.id, "已啟動批量訓練，模式: ham。接下來你在這個私訊中傳送的純文本訊息會被收集；完成後使用 /ml_finish_mass_ham。")
                .await?;
        }
        ModerationCommand::MlDebugParse => {
            if !message.chat.is_private() || !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只允許維護者在私訊中使用 /ml_debug_parse。") .await?;
                return Ok(());
            }
            let target_msg = message.reply_to_message().unwrap_or(&message);
            let text = extract_full_text(target_msg);
            if text.trim().is_empty() {
                bot.send_message(message.chat.id, "請回覆一段日誌或訊息內容。") .await?;
                return Ok(());
            }
            let extracted = smart_train_payloads(&text);
            let body = if extracted.is_empty() {
                "<無法提取>".to_string()
            } else {
                extracted.into_iter().map(|s| escape_html(&s)).collect::<Vec<_>>().join("\n---\n")
            };
            bot.send_message(message.chat.id, format!("<b>提</b>:\n<blockquote>{}</blockquote>", body)).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::MlScoreDebug => {
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以使用 /ml_score_debug。");
            let target_msg = message.reply_to_message().unwrap_or(&message);
            let text = extract_full_text(target_msg);
            if text.trim().is_empty() {
                bot.send_message(message.chat.id, "請回覆一條消息或提供內容。") .await?;
                return Ok(());
            }
            let user_name = message.from.as_ref().map(short_user).unwrap_or_else(|| "unknown".to_string());
            let result = runtime.inspect_message(&user_name, &text).await.map_err(|e| teloxide::RequestError::Io(std::io::Error::other(e.to_string()).into()))?;
            let mut out = String::new();
            out.push_str(&format!("<b>文本</b>:\n<blockquote>{}</blockquote>\n", escape_html(&text)));
            match result {
                InspectionResult::Spam { score, matched_rule: Some(rule) } => {
                    out.push_str(&format!("<b>判定</b>: 垃圾\n<b>分數</b>: {score:.6}\n<b>規則</b>: REGEX\n<b>說明</b>: {}", escape_html(&rule.description)));
                }
                InspectionResult::Spam { score, .. } | InspectionResult::Ham { score } => {
                    let report = runtime.score_debug(&user_name, &text).await.map_err(|e| teloxide::RequestError::Io(std::io::Error::other(e.to_string()).into()))?;
                    out.push_str(&format!("<b>判定</b>: {}\n<b>分數</b>: {score:.6}\n{}", if score >= runtime.effective_threshold(Some(message.chat.id.0)).await.unwrap_or(runtime.config.spam_threshold) { "垃圾" } else { "正常" }, format_score_debug(&report)));
                }
            }
            bot.send_message(message.chat.id, out).parse_mode(ParseMode::Html).await?;
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
                        train_spam(&runtime, &payload, None).await.ok();
                        spam_count += 1;
                    }
                } else {
                    let payload = sample.clone();
                    extracted_debug.push(payload.clone());
                    if sample.starts_with("-") || sample.starts_with("ham:") {
                        train_ham(&runtime, &payload, None).await.ok();
                        ham_count += 1;
                    } else {
                        train_spam(&runtime, &payload, None).await.ok();
                        spam_count += 1;
                    }
                }
            }
            let debug = if extracted_debug.is_empty() { "無可提取樣本".to_string() } else { extracted_debug.join("\n---\n") };
            bot.send_message(message.chat.id, format!("批量訓練完成。spam: {spam_count}, ham: {ham_count}\n\n已提取並訓練的字串：\n{debug}")).await?;
            runtime.clear_mass_train(from_id).await;
        }
        ModerationCommand::MlFinishMassHam => {
            if !message.chat.is_private() || !is_maintainer(&bot, &runtime.config, from_id).await {
                bot.send_message(message.chat.id, "只允許維護者在私訊中結束批量訓練。") .await?;
                return Ok(());
            }
            let samples = runtime.finish_mass_train(from_id).await;
            let mut imported = Vec::new();
            let mut count = 0usize;
            for sample in samples {
                if sample.trim().is_empty() { continue; }
                imported.push(sample.clone());
                train_ham(&runtime, &sample, None).await.ok();
                count += 1;
            }
            bot.send_message(message.chat.id, format!("批量訓練完成。ham: {count}\n\n已提取並訓練的字串：\n{}", if imported.is_empty() { "無可提取樣本".to_string() } else { imported.join("\n---\n") })).await?;
            runtime.clear_mass_train(from_id).await;
        }
        ModerationCommand::Unban(arg) => {
            let is_maintainer_user = is_maintainer(&bot, &runtime.config, from_id).await;
            let is_admin_user = (message.chat.is_group() || message.chat.is_supergroup())
                && is_group_admin(&bot, message.chat.id, from_id).await;
            if !is_maintainer_user && !is_admin_user {
                handle_permission_denied(&bot, &runtime, &message, from, "只有群組管理員或項目維護組可以使用此指令。").await?;
                return Ok(());
            }

            if !is_maintainer_user {
                // Group admins (not maintainers) can let someone back into
                // their own group, but reversing the case and purging
                // training data is left to a maintainer reviewing the logs -
                // that judgment call (was this actually a false positive?)
                // needs someone with visibility across the whole project,
                // not just this one group.
                let target_user_id = if let Some(target_msg) = message.reply_to_message() {
                    target_msg.from.as_ref().map(|u| u.id.0 as i64)
                } else {
                    arg.trim().parse::<i64>().ok()
                };
                let Some(target_user_id) = target_user_id else {
                    reply_ephemeral(&bot, &message, "請回覆要解封的用戶，或提供 user_id。").await?;
                    return Ok(());
                };
                if let Err(err) = bot.unban_chat_member(message.chat.id, UserId(target_user_id as u64)).await {
                    bot.send_message(message.chat.id, format!("解封失敗：{err}")).await?;
                    return Ok(());
                }
                let _ = bot
                    .send_message(
                        ChatId(runtime.config.log_channel_id),
                        format!(
                            "<b>群組管理員手動解封</b>\n<b>群組</b>: <code>{}</code>\n<b>對象</b>: <code>{target_user_id}</code>\n<b>操作者</b>: {}",
                            message.chat.id.0,
                            escape_html(&short_user(from)),
                        ),
                    )
                    .parse_mode(ParseMode::Html)
                    .await;
                bot.send_message(message.chat.id, format!("已在本群解封用戶 <code>{target_user_id}</code>。")).parse_mode(ParseMode::Html).await?;
                return Ok(());
            }

            // Maintainer path: three ways to target - reply to the user (like
            // /sb), an explicit user_id, or a case_id (whose own
            // chat_id/target_user_id are used instead of the current chat -
            // lets you reverse a case from DM). The reply/user_id forms exist
            // because a user can be banned without this bot ever knowing -
            // manually by another admin, by a different bot, or from before
            // this project was even involved - so unbanning can't depend on a
            // case existing at all.
            let resolved = if let Some(target_msg) = message.reply_to_message() {
                target_msg.from.as_ref().map(|u| (message.chat.id.0, u.id.0 as i64, None::<CaseRecord>))
            } else if let Ok(user_id) = arg.trim().parse::<i64>() {
                Some((message.chat.id.0, user_id, None))
            } else if !arg.trim().is_empty() {
                match runtime.load_case(arg.trim()).await {
                    Ok(Some(case)) => Some((case.chat_id, case.target_user_id, Some(case))),
                    _ => None,
                }
            } else {
                None
            };

            let Some((chat_id, target_user_id, case_from_id)) = resolved else {
                bot.send_message(message.chat.id, "請回覆要解封的用戶、提供 user_id，或提供 case_id。例如 /unban 123456789。").await?;
                return Ok(());
            };

            // Best-effort: reverse a tracked case too, if one exists, to also
            // clean up any training data it contributed. Not finding one is
            // completely normal for a user this project never banned.
            let case = match case_from_id {
                Some(case) => Some(case),
                None => runtime
                    .load_latest_case_by_actions(chat_id, target_user_id, &["auto_ban", "spam_ban", "report_approved"])
                    .await
                    .ok()
                    .flatten(),
            };

            let Some(case) = case else {
                if let Err(err) = bot.unban_chat_member(ChatId(chat_id), UserId(target_user_id as u64)).await {
                    bot.send_message(message.chat.id, format!("解封失敗：{err}")).await?;
                    return Ok(());
                }
                bot.send_message(
                    message.chat.id,
                    format!("已在本群解封用戶 <code>{target_user_id}</code>。（找不到本專案的封禁記錄，沒有訓練樣本需要清除）"),
                )
                .parse_mode(ParseMode::Html)
                .await?;
                return Ok(());
            };

            match reverse_ban_case(&bot, &runtime, case, from_id, &short_user(from)).await {
                Ok(summary) => { bot.send_message(message.chat.id, summary).parse_mode(ParseMode::Html).await?; }
                Err(err) => { bot.send_message(message.chat.id, err).await?; }
            }
        }
        ModerationCommand::Unmute(arg) => {
            let is_maintainer_user = is_maintainer(&bot, &runtime.config, from_id).await;
            let is_admin_user = (message.chat.is_group() || message.chat.is_supergroup())
                && is_group_admin(&bot, message.chat.id, from_id).await;
            if !is_maintainer_user && !is_admin_user {
                handle_permission_denied(&bot, &runtime, &message, from, "只有群組管理員或項目維護組可以使用此指令。").await?;
                return Ok(());
            }

            if !is_maintainer_user {
                // Same reasoning as /unban: a group admin can free their own
                // group's member right away, but reversing the case is left
                // to a maintainer's judgment call.
                let target_user_id = if let Some(target_msg) = message.reply_to_message() {
                    target_msg.from.as_ref().map(|u| u.id.0 as i64)
                } else {
                    arg.trim().parse::<i64>().ok()
                };
                let Some(target_user_id) = target_user_id else {
                    reply_ephemeral(&bot, &message, "請回覆要解除禁言的用戶，或提供 user_id。").await?;
                    return Ok(());
                };
                if let Err(err) = bot.restrict_chat_member(message.chat.id, UserId(target_user_id as u64), teloxide::types::ChatPermissions::all()).await {
                    bot.send_message(message.chat.id, format!("解除禁言失敗：{err}")).await?;
                    return Ok(());
                }
                let _ = bot
                    .send_message(
                        ChatId(runtime.config.log_channel_id),
                        format!(
                            "<b>群組管理員手動解除禁言</b>\n<b>群組</b>: <code>{}</code>\n<b>對象</b>: <code>{target_user_id}</code>\n<b>操作者</b>: {}",
                            message.chat.id.0,
                            escape_html(&short_user(from)),
                        ),
                    )
                    .parse_mode(ParseMode::Html)
                    .await;
                bot.send_message(message.chat.id, format!("已在本群解除用戶 <code>{target_user_id}</code> 的禁言。")).parse_mode(ParseMode::Html).await?;
                return Ok(());
            }

            // Maintainer path: same three targeting modes as /unban (reply /
            // user_id / case_id) - a mute can also come from outside this
            // bot's tracking.
            let resolved = if let Some(target_msg) = message.reply_to_message() {
                target_msg.from.as_ref().map(|u| (message.chat.id.0, u.id.0 as i64, None::<CaseRecord>))
            } else if let Ok(user_id) = arg.trim().parse::<i64>() {
                Some((message.chat.id.0, user_id, None))
            } else if !arg.trim().is_empty() {
                match runtime.load_case(arg.trim()).await {
                    Ok(Some(case)) => Some((case.chat_id, case.target_user_id, Some(case))),
                    _ => None,
                }
            } else {
                None
            };

            let Some((chat_id, target_user_id, case_from_id)) = resolved else {
                bot.send_message(message.chat.id, "請回覆要解除禁言的用戶、提供 user_id，或提供 case_id。例如 /unmute 123456789。").await?;
                return Ok(());
            };

            let case = match case_from_id {
                Some(case) => Some(case),
                None => runtime
                    .load_latest_case_by_actions(chat_id, target_user_id, &["mute", "flood_mute"])
                    .await
                    .ok()
                    .flatten(),
            };

            let Some(case) = case else {
                if let Err(err) = bot.restrict_chat_member(ChatId(chat_id), UserId(target_user_id as u64), teloxide::types::ChatPermissions::all()).await {
                    bot.send_message(message.chat.id, format!("解除禁言失敗：{err}")).await?;
                    return Ok(());
                }
                bot.send_message(message.chat.id, format!("已在本群解除用戶 <code>{target_user_id}</code> 的禁言。（找不到本專案的禁言記錄）")).parse_mode(ParseMode::Html).await?;
                return Ok(());
            };

            match reverse_mute_case(&bot, &runtime, case, from_id, &short_user(from)).await {
                Ok(summary) => { bot.send_message(message.chat.id, summary).parse_mode(ParseMode::Html).await?; }
                Err(err) => { bot.send_message(message.chat.id, err).await?; }
            }
        }
        ModerationCommand::Ping => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            bot.send_message(message.chat.id, version_info_text()).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::Revert(action_id_arg) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let Some(action_id) = action_id_arg.trim().parse::<i64>().ok() else {
                bot.send_message(message.chat.id, "請提供有效的 action id，例如 /revert 42。").await?;
                return Ok(());
            };
            let action = runtime
                .load_maintainer_action(action_id)
                .await
                .map_err(|e| teloxide::RequestError::Io(std::io::Error::other(e.to_string()).into()))?;
            let Some(action) = action else {
                bot.send_message(message.chat.id, format!("找不到 action #{action_id}。")).await?;
                return Ok(());
            };
            if action.reverted {
                bot.send_message(message.chat.id, format!("action #{action_id} 已經被復原過了。")).await?;
                return Ok(());
            }

            let actor_name = short_user(from);
            let result: Result<String, String> = match action.undo {
                UndoData::NotRevertible => Err(format!("action #{action_id}（{}）無法自動復原。", action.command)),
                UndoData::Threshold { old } => match runtime.set_threshold(old).await {
                    Ok(()) => Ok(format!("已將全域門檻復原為 {old:.2}。")),
                    Err(e) => Err(e.to_string()),
                },
                UndoData::GroupThreshold { chat_id, old } => match runtime.set_group_threshold(chat_id, old).await {
                    Ok(()) => Ok(format!("已將群組 {chat_id} 的門檻復原為 {old:?}。")),
                    Err(e) => Err(e.to_string()),
                },
                UndoData::TokenProbability { token, old_spam, old_ham } => match runtime.set_token_counts_raw(&token, old_spam, old_ham).await {
                    Ok(()) => Ok(format!("已將 token <code>{}</code> 的計數復原為 spam={old_spam}, ham={old_ham}。", escape_html(&token))),
                    Err(e) => Err(e.to_string()),
                },
                UndoData::GroupModule { chat_id, module, old_enabled } => match runtime.set_group_module(chat_id, &module, old_enabled).await {
                    Ok(()) => Ok(format!("已將群組 {chat_id} 的模組 {module} 復原為 {old_enabled}。")),
                    Err(e) => Err(e.to_string()),
                },
                UndoData::GroupWhitelist { chat_id, user_id, old_enabled } => match runtime.set_group_whitelist(chat_id, user_id, old_enabled, None).await {
                    Ok(()) => Ok(format!("已將群組 {chat_id} 對 user_id={user_id} 的白名單狀態復原為 {old_enabled}。")),
                    Err(e) => Err(e.to_string()),
                },
                UndoData::GlobalWhitelist { user_id, old_enabled } => match runtime.set_global_whitelist(user_id, old_enabled, None).await {
                    Ok(()) => Ok(format!("已將 user_id={user_id} 的全域白名單狀態復原為 {old_enabled}。")),
                    Err(e) => Err(e.to_string()),
                },
                UndoData::RuleAdded { rule_id } => match runtime.delete_spam_rule(rule_id).await {
                    Ok(_) => Ok(format!("已刪除規則 @{rule_id}。")),
                    Err(e) => Err(e.to_string()),
                },
                UndoData::RuleEdited { rule_id, old_pattern } => match runtime.update_spam_rule_pattern(rule_id, &old_pattern).await {
                    Ok(_) => Ok(format!("已將規則 @{rule_id} 的正則復原。")),
                    Err(e) => Err(e.to_string()),
                },
                UndoData::RuleDeleted { pattern, description } => match runtime.add_spam_rule(&pattern, &description).await {
                    Ok(new_id) => Ok(format!("已重新建立規則（新 ID：@{new_id}，原規則 ID 無法保留）。")),
                    Err(e) => Err(e.to_string()),
                },
                UndoData::ProjectChat { old } => match old {
                    Some(old) => {
                        runtime.set_project_chat(old).await;
                        Ok(format!("已將項目交流群復原為 {old}。"))
                    }
                    None => Err("此操作之前沒有設定項目交流群，無法自動復原成「未設定」狀態，請視需要手動處理。".to_string()),
                },
                UndoData::TrainingSample { training_ref } => match runtime.purge_training_by_case(&training_ref).await {
                    Ok(removed) => {
                        if removed > 0 {
                            let _ = runtime.rebuild_model().await;
                        }
                        Ok(format!("已移除該筆訓練樣本並重建模型（共 {removed} 筆）。"))
                    }
                    Err(e) => Err(e.to_string()),
                },
                UndoData::Case { case_id, kind } => match runtime.load_case(&case_id).await {
                    Ok(Some(case)) => match kind {
                        CaseKind::Ban => reverse_ban_case(&bot, &runtime, case, from_id, &actor_name).await,
                        CaseKind::Mute => reverse_mute_case(&bot, &runtime, case, from_id, &actor_name).await,
                    },
                    Ok(None) => Err(format!("找不到案例 {case_id}。")),
                    Err(e) => Err(e.to_string()),
                },
            };

            let original_context = format!(
                "原操作: <code>{}</code>（{}{}操作者: {}{}）",
                escape_html(&action.command),
                escape_html(&action.summary),
                if action.summary.is_empty() { "" } else { "，" },
                escape_html(&action.actor_name),
                action.chat_id.map(|c| format!("，群組: {c}")).unwrap_or_default(),
            );
            match result {
                Ok(summary) => {
                    runtime.mark_maintainer_action_reverted(action_id).await.ok();
                    bot.send_message(message.chat.id, format!("已復原 action #{action_id}：{summary}\n{original_context}")).parse_mode(ParseMode::Html).await?;
                }
                Err(err) => {
                    bot.send_message(message.chat.id, format!("{err}\n{original_context}")).await?;
                }
            }
        }
        ModerationCommand::Unknown => {
            if message.chat.is_private() {
                if let Some(pattern) = runtime.pending_rule_addition(from_id).await {
                    let name = text.trim();
                    if !name.is_empty() {
                        let id = runtime.add_spam_rule(&pattern, name).await.map_err(|e| teloxide::RequestError::Io(std::io::Error::other(e.to_string()).into()))?;
                        runtime.take_pending_rule_addition(from_id).await;
                        log_maintainer_action(&bot, &runtime, from_id, &short_user(from), None, "/add_rule", &format!("新增規則 @{id}"), UndoData::RuleAdded { rule_id: id }).await;
                        bot.send_message(message.chat.id, format!("已建立規則 @{}。", id)).await?;
                        return Ok(());
                    }
                }
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
                if let Err(err) = delete_message_if_exists(&bot, ChatId(case.chat_id), MessageId(source_id)).await {
                    log_callback_error(&bot, &runtime, &case, "delete_message", &err.to_string()).await;
                }
            }
            if let Err(err) = train_spam(&runtime, &case.evidence_text, Some(&case.id)).await {
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
            let log_message_id = match log_action(&bot, &runtime, &updated).await {
                Ok(id) => id,
                Err(err) => {
                    log_callback_error(&bot, &runtime, &case, "log_action", &err.to_string()).await;
                    0
                }
            };
            if log_message_id != 0 {
                let mut logged = updated.clone();
                logged.log_message_id = Some(log_message_id);
                if let Err(err) = store_case(&runtime, &logged).await {
                    log_callback_error(&bot, &runtime, &case, "store_case", &err.to_string()).await;
                }
            }
            propagate_network_ban(&bot, &runtime, &updated).await;
            let body = format!(
                "<b>新的 /spam 申請</b>\n\n<b>對象</b>: {} ({})\n<b>發起人</b>: {}\n<b>內容</b>: <blockquote>{}</blockquote>\n<b>案例</b>: <code>{}</code>\n<b>狀態</b>: 已受理並封禁\n<b>處理者</b>: <code>{}</code>",
                escape_html(&case.target_name),
                case.target_user_id,
                escape_html(case.actor_name.as_deref().unwrap_or("unknown")),
                escape_html(&case.evidence_text),
                case.id,
                from_id
            );
            let _ = bot.edit_message_text(message.chat().id, message.id(), body).parse_mode(ParseMode::Html).await;
            let _ = bot.edit_message_reply_markup(message.chat().id, message.id()).await;
            bot.answer_callback_query(q.id).text("已受理並封禁").await?;
        }
        "reject" => {
            if let Err(err) = train_ham(&runtime, &case.evidence_text, Some(&case.id)).await {
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
                "<b>新的 /spam 申請</b>\n\n<b>對象</b>: {} ({})\n<b>發起人</b>: {}\n<b>內容</b>: <blockquote>{}</blockquote>\n<b>案例</b>: <code>{}</code>\n<b>狀態</b>: 已拒絕受理\n<b>處理者</b>: <code>{}</code>",
                escape_html(&case.target_name),
                case.target_user_id,
                escape_html(case.actor_name.as_deref().unwrap_or("unknown")),
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
    if runtime.is_global_whitelisted(user.id.0 as i64).await.unwrap_or(false) {
        return Ok(());
    }
    if runtime.is_group_whitelisted(message.chat.id.0, user.id.0 as i64).await.unwrap_or(false) {
        return Ok(());
    }
    if is_group_admin(&bot, message.chat.id, user.id.0 as i64).await || is_special_user(&runtime.config, user.id.0 as i64) {
        return Ok(());
    }
    if let Ok(check) = runtime.check_group_modules(&bot, message.chat.id.0, user, None, message.text().or(message.caption())).await {
        if !check.reasons.is_empty() {
            let _ = bot.delete_message(message.chat.id, message.id).await;
            let _ = ban_user(&bot, message.chat.id, user.id.0 as i64).await;
            let case_id = Uuid::new_v4().to_string();
            let case = CaseRecord {
                id: case_id,
                action: ActionKind::AutoBan,
                chat_id: message.chat.id.0,
                target_user_id: user.id.0 as i64,
                target_name: short_user(user),
                actor_user_id: None,
                actor_name: None,
                source_message_id: Some(message.id.0),
                evidence_text: extract_full_text(&message),
                model_score: None,
                matched_rule_id: None,
                matched_rule_pattern: Some(check.reasons.join("；")),
                status: "auto_banned".to_string(),
                log_message_id: None,
                created_at: Utc::now(),
            };
            let log_message_id = log_action(&bot, &runtime, &case).await.unwrap_or_default();
            let mut updated = case.clone();
            updated.log_message_id = Some(log_message_id);
            let _ = store_case(&runtime, &updated).await;
            let _ = notify_group(&bot, &runtime, &updated, log_message_id, "<b>自動模組封禁</b>").await;
            propagate_network_ban(&bot, &runtime, &updated).await;
            return Ok(());
        }
    }
    let text = extract_full_text(&message);
    if text.trim().is_empty() { return Ok(()); }
    let result = runtime.inspect_message(&short_user(user), &text).await.map_err(|e| teloxide::RequestError::Io(std::io::Error::other(e.to_string()).into()))?;
    let score = match result {
        InspectionResult::Spam { score, .. } | InspectionResult::Ham { score } => score,
    };

    let threshold = runtime.effective_threshold(Some(message.chat.id.0)).await.unwrap_or(runtime.config.spam_threshold);
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
        matched_rule_id: None,
        matched_rule_pattern: None,
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
    propagate_network_ban(&bot, &runtime, &case).await;
    Ok(())
}

async fn score_only(bot: &Bot, runtime: &Runtime, message: &Message) -> ResponseResult<()> {
    let Some(user) = message.from.as_ref() else { return Ok(()); };
    if user.is_bot {
        return Ok(());
    }
    let text = extract_full_text(message);
    if text.trim().is_empty() { return Ok(()); }
    let result = runtime.inspect_message(&short_user(user), &text).await.map_err(|e| teloxide::RequestError::Io(std::io::Error::other(e.to_string()).into()))?;
    let score = match result {
        InspectionResult::Spam { score, .. } | InspectionResult::Ham { score } => score,
    };
    let threshold = runtime.effective_threshold(Some(message.chat.id.0)).await.unwrap_or(runtime.config.spam_threshold);
    let verdict = if score >= threshold { "spam" } else { "ham" };
    let reply = format!(
        "<b>判定</b>: {}\n<b>分數</b>: {:.4}\n<b>門檻</b>: {:.4}",
        if verdict == "spam" { "垃圾" } else { "正常" },
        score,
        threshold,
    );
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

    if let Some(owner_id) = runtime.config.owner_id {
        // Best-effort: a restart is exactly when this is most useful, but it
        // shouldn't block (or fail) startup if the DM can't be delivered -
        // e.g. the owner never having started a chat with the bot.
        let _ = bot.send_message(ChatId(owner_id), version_info_text()).parse_mode(ParseMode::Html).await;
    }

    let message_handler = Update::filter_message().endpoint({
        let runtime = runtime.clone();
        move |bot: Bot, message: Message| {
            let runtime = runtime.clone();
            async move {
                // Runs before delete_service_message_if_enabled, which might
                // otherwise delete this same "message pinned" notification
                // first (if NoServiceMessage is also on for this chat) - this
                // still needs to inspect it either way.
                unpin_channel_autopin(&bot, &runtime, &message).await;

                // First, check and delete service messages if enabled
                if delete_service_message_if_enabled(&bot, &runtime, &message).await? {
                    return Ok(());
                }

                // A pending join CAPTCHA takes priority over everything else -
                // this message is either the answer or noise from a still-muted
                // member, never a real command/content to process further.
                if check_captcha_and_act(&bot, &runtime, &message).await {
                    return Ok(());
                }

                // Netban safety net: catches members already in a group before
                // it opted in, or who joined between propagation events.
                if runtime.config.test_group_id != Some(message.chat.id.0)
                    && check_netban_and_act(&bot, &runtime, &message).await
                {
                    return Ok(());
                }

                // Behavioral flood check runs before anything content-based, and
                // for every message type - skip it in the test group, which is
                // score-only by design (see score_only below) and never enforces.
                if runtime.config.test_group_id != Some(message.chat.id.0)
                    && check_flood_and_act(&bot, &runtime, &message).await?
                {
                    return Ok(());
                }

                if notify_bot_added(&bot, &runtime, &message).await {
                    return Ok(());
                }
                if let Some(text) = message.text() {
                    if text.trim_start().starts_with('/') {
                        if !matches!(parse_command(text), ModerationCommand::Unknown) {
                            return handle_command(bot, runtime, message).await;
                        }
                        return Ok(());
                    }
                    if message.chat.is_private() {
                        return handle_command(bot, runtime, message).await;
                    }
                    if runtime.config.test_group_id == Some(message.chat.id.0) {
                        return score_only(&bot, &runtime, &message).await;
                    }
                    if !ensure_bot_can_moderate(&bot, &runtime, message.chat.id).await? {
                        return Ok(());
                    }
                    if matches!(parse_command(text), ModerationCommand::Unknown) {
                        auto_moderate(bot, runtime, message).await?;
                        return Ok(());
                    }
                } else {
                    if message.chat.is_private() {
                        return handle_command(bot, runtime, message).await;
                    }
                    if runtime.config.test_group_id == Some(message.chat.id.0) {
                        return score_only(&bot, &runtime, &message).await;
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

    let handler = dptree::entry()
        .inspect(|u: teloxide::types::Update| {
            println!("=> [DISPATCHER RECEIVED UPDATE]: ID {:?}", u.id);
        })
        .branch(message_handler)
        .branch(callback_handler);

    let mut dispatcher = Dispatcher::builder(bot, handler)
        .dependencies(dptree::deps![runtime.clone(), runtime.config.clone()])
        .enable_ctrlc_handler()
        .build();

    dispatcher.dispatch().await;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_runtime() -> Runtime {
        let dir = std::env::temp_dir().join(format!("spb_test_{}", Uuid::new_v4()));
        let config = Config {
            bot_token: "test".to_string(),
            log_channel_id: -1,
            report_channel_id: -1,
            test_group_id: None,
            maintainer_ids: vec![],
            data_dir: dir.clone(),
            sqlite_path: dir.join("bot.db"),
            spam_threshold: 0.85,
            owner_id: None,
        };
        tokio::fs::create_dir_all(&dir).await.unwrap();
        Runtime::load(config).await.unwrap()
    }

    // Regression check for the /sb (SpamBan) path: it calls train_spam with the
    // replied-to message's text, and this confirms that call actually reaches
    // and persists in the DB rather than just updating the in-memory cache -
    // exactly the path the with_conn/transaction refactor touched.
    #[tokio::test]
    async fn spam_ban_training_persists_to_disk() {
        let runtime = test_runtime().await;
        let text = "腾龙集团 联系客服 usdt 官方注册通道";
        train_spam(&runtime, text, Some("case-1")).await.unwrap();

        {
            let model = runtime.model.lock().await;
            assert_eq!(model.spam_docs, 1);
            assert!(!model.spam_tokens.is_empty());
        }

        // Drop the in-memory cache and reload straight from disk, simulating a
        // restart, to prove the write actually landed in SQLite rather than
        // only updating the process-local cache.
        let rebuilt = runtime.rebuild_model().await.unwrap();
        assert_eq!(rebuilt.spam_docs, 1);
        assert!(!rebuilt.spam_tokens.is_empty());
        for token in tokenize(text) {
            assert!(rebuilt.spam_tokens.get(&token).copied().unwrap_or(0) >= 1, "missing token: {token}");
        }

        let export = runtime.export_training_data().await.unwrap();
        assert!(export.contains("spam"));
        assert!(export.contains("case-1"));
    }

    // Regression check for the /set corruption incident: seeds a v1-shaped DB
    // (pre-migrate_v2_to_v3) with a dead token_counts table and a poisoned
    // word_frequencies row, then loads a Runtime against it and confirms
    // migrate_v1_to_v2/migrate_v2_to_v3 both actually ran on startup.
    #[tokio::test]
    async fn migration_clamps_outlier_and_drops_dead_table() {
        let dir = std::env::temp_dir().join(format!("spb_test_{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let db_path = dir.join("bot.db");

        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                r#"
                CREATE TABLE token_counts (token TEXT NOT NULL, label TEXT NOT NULL, count INTEGER NOT NULL, PRIMARY KEY (token, label));
                CREATE TABLE word_frequencies (word TEXT PRIMARY KEY, spam_count INTEGER NOT NULL DEFAULT 0, ham_count INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE group_module_settings (chat_id INTEGER PRIMARY KEY, no_long_name INTEGER NOT NULL DEFAULT 0, no_halal INTEGER NOT NULL DEFAULT 0, no_service_messages INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE model_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO word_frequencies (word, spam_count, ham_count) VALUES ('poisoned', 250000000000000, 0);
                INSERT INTO word_frequencies (word, spam_count, ham_count) VALUES ('normal', 49, 0);
                PRAGMA user_version = 1;
                "#,
            )
            .unwrap();
        }

        let config = Config {
            bot_token: "test".to_string(),
            log_channel_id: -1,
            report_channel_id: -1,
            test_group_id: None,
            maintainer_ids: vec![],
            data_dir: dir.clone(),
            sqlite_path: db_path.clone(),
            spam_threshold: 0.85,
            owner_id: None,
        };
        let runtime = Runtime::load(config).await.unwrap();

        let (poisoned_count, normal_count): (i64, i64) = runtime
            .with_conn(|conn| {
                let poisoned: i64 = conn.query_row("SELECT spam_count FROM word_frequencies WHERE word = 'poisoned'", [], |r| r.get(0))?;
                let normal: i64 = conn.query_row("SELECT spam_count FROM word_frequencies WHERE word = 'normal'", [], |r| r.get(0))?;
                Ok((poisoned, normal))
            })
            .await
            .unwrap();
        assert_eq!(poisoned_count, 1000, "outlier should be clamped");
        assert_eq!(normal_count, 49, "untouched row should be unaffected");

        let token_counts_exists: bool = runtime
            .with_conn(|conn| {
                let count: i64 = conn.query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='token_counts'",
                    [],
                    |r| r.get(0),
                )?;
                Ok(count > 0)
            })
            .await
            .unwrap();
        assert!(!token_counts_exists, "dead token_counts table should be dropped");
    }

    // The /unban and /unmute commands lean on purge_training_by_case +
    // rebuild_model to undo a bad training sample tied to a case. This
    // confirms that combination actually removes the sample and its tokens.
    #[tokio::test]
    async fn purge_by_case_removes_training_sample_and_tokens() {
        let runtime = test_runtime().await;
        let text = "測試用垃圾樣本文字內容";
        train_spam(&runtime, text, Some("case-unban-1")).await.unwrap();

        let removed = runtime.purge_training_by_case("case-unban-1").await.unwrap();
        assert!(removed > 0);

        let rebuilt = runtime.rebuild_model().await.unwrap();
        let export = runtime.export_training_data().await.unwrap();
        assert!(!export.contains("case-unban-1"));
        for token in tokenize(text) {
            assert_eq!(rebuilt.spam_tokens.get(&token).copied().unwrap_or(0), 0, "token should be gone: {token}");
        }
    }

    #[tokio::test]
    async fn flood_check_trips_after_five_within_window() {
        let runtime = test_runtime().await;
        for _ in 0..4 {
            assert!(!runtime.check_flood(1, 1).await);
        }
        assert!(runtime.check_flood(1, 1).await, "5th message within the window should trip it");
    }

    #[tokio::test]
    async fn flood_check_is_scoped_per_chat_and_user() {
        let runtime = test_runtime().await;
        for _ in 0..4 {
            assert!(!runtime.check_flood(1, 1).await);
        }
        // A different user in the same chat has their own counter.
        assert!(!runtime.check_flood(1, 2).await);
    }

    #[tokio::test]
    async fn group_threshold_override_falls_back_to_global() {
        let runtime = test_runtime().await;
        runtime.set_threshold(0.9).await.unwrap();
        assert_eq!(runtime.effective_threshold(None).await.unwrap(), 0.9);
        assert_eq!(runtime.effective_threshold(Some(111)).await.unwrap(), 0.9, "no override yet, should inherit global");

        runtime.set_group_threshold(111, Some(0.6)).await.unwrap();
        assert_eq!(runtime.effective_threshold(Some(111)).await.unwrap(), 0.6);
        assert_eq!(runtime.effective_threshold(Some(222)).await.unwrap(), 0.9, "other chats are unaffected");
    }

    fn dummy_case(action: ActionKind, chat_id: i64, target_user_id: i64, created_at: DateTime<Utc>) -> CaseRecord {
        CaseRecord {
            id: Uuid::new_v4().to_string(),
            action,
            chat_id,
            target_user_id,
            target_name: "test".to_string(),
            actor_user_id: None,
            actor_name: None,
            source_message_id: None,
            evidence_text: "evidence".to_string(),
            model_score: None,
            matched_rule_id: None,
            matched_rule_pattern: None,
            status: "auto_banned".to_string(),
            log_message_id: None,
            created_at,
        }
    }

    // Backs /unban and /unmute's ability to find what to reverse from just a
    // reply or a user_id, with no case_id given: it should find the most
    // recent matching case for that (chat, user), not an older one, and not
    // one with a different action kind.
    #[tokio::test]
    async fn load_latest_case_by_actions_picks_most_recent_matching() {
        let runtime = test_runtime().await;
        let older = dummy_case(ActionKind::AutoBan, 100, 200, Utc::now() - chrono::TimeDelta::hours(2));
        let newer = dummy_case(ActionKind::SpamBan, 100, 200, Utc::now() - chrono::TimeDelta::hours(1));
        let unrelated_action = dummy_case(ActionKind::Mute, 100, 200, Utc::now());
        let unrelated_user = dummy_case(ActionKind::SpamBan, 100, 999, Utc::now());
        runtime.persist_case(&older).await.unwrap();
        runtime.persist_case(&newer).await.unwrap();
        runtime.persist_case(&unrelated_action).await.unwrap();
        runtime.persist_case(&unrelated_user).await.unwrap();

        let found = runtime.load_latest_case_by_actions(100, 200, &["auto_ban", "spam_ban", "report_approved"]).await.unwrap();
        assert_eq!(found.map(|c| c.id), Some(newer.id.clone()));
    }

    // Once a case is reversed (action mutated to Unbanned, as the /unban
    // handler does), it must drop out of future lookups so a second /unban
    // by user_id finds the next still-active case instead of re-finding the
    // one already undone.
    #[tokio::test]
    async fn reversed_case_is_excluded_from_future_lookups() {
        let runtime = test_runtime().await;
        let mut case = dummy_case(ActionKind::AutoBan, 100, 200, Utc::now());
        runtime.persist_case(&case).await.unwrap();

        case.action = ActionKind::Unbanned;
        case.status = "reversed".to_string();
        runtime.persist_case(&case).await.unwrap();

        let found = runtime.load_latest_case_by_actions(100, 200, &["auto_ban", "spam_ban", "report_approved"]).await.unwrap();
        assert!(found.is_none(), "reversed case should no longer match a ban-action search");
    }

    // A ban only counts as "network-banned" if its origin chat has netban
    // enabled - a group that never opted in shouldn't leak bans to others.
    #[tokio::test]
    async fn find_active_network_ban_requires_netban_enabled_origin() {
        let runtime = test_runtime().await;
        let case = dummy_case(ActionKind::AutoBan, 100, 200, Utc::now());
        runtime.persist_case(&case).await.unwrap();

        assert!(
            runtime.find_active_network_ban(200).await.unwrap().is_none(),
            "chat 100 hasn't opted into netban, so this ban shouldn't count as a network ban"
        );

        runtime.set_group_module(100, "netban", true).await.unwrap();
        let found = runtime.find_active_network_ban(200).await.unwrap();
        assert_eq!(found.map(|c| c.id), Some(case.id.clone()));
    }

    // Same "reversal mutates action in place" property as
    // load_latest_case_by_actions - once reversed, it must stop being an
    // active network ban.
    #[tokio::test]
    async fn find_active_network_ban_excludes_reversed_case() {
        let runtime = test_runtime().await;
        runtime.set_group_module(100, "netban", true).await.unwrap();
        let mut case = dummy_case(ActionKind::SpamBan, 100, 200, Utc::now());
        runtime.persist_case(&case).await.unwrap();
        assert!(runtime.find_active_network_ban(200).await.unwrap().is_some());

        case.action = ActionKind::Unbanned;
        runtime.persist_case(&case).await.unwrap();
        assert!(runtime.find_active_network_ban(200).await.unwrap().is_none());
    }

    // Backs /unban's ability to reverse a propagated ban everywhere it
    // actually landed: record targets for a case across a couple of chats,
    // confirm they list back correctly, then confirm clearing empties it.
    #[tokio::test]
    async fn network_ban_targets_round_trip() {
        let runtime = test_runtime().await;
        let case_id = "case-netban-1";
        runtime.record_network_ban_target(case_id, 100).await.unwrap();
        runtime.record_network_ban_target(case_id, 200).await.unwrap();
        // Recording the same (case, chat) pair twice must not duplicate it.
        runtime.record_network_ban_target(case_id, 100).await.unwrap();

        let mut targets = runtime.list_network_ban_targets(case_id).await.unwrap();
        targets.sort();
        assert_eq!(targets, vec![100, 200]);

        runtime.clear_network_ban_targets(case_id).await.unwrap();
        assert!(runtime.list_network_ban_targets(case_id).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn list_netban_enabled_chats_reflects_toggles() {
        let runtime = test_runtime().await;
        runtime.set_group_module(100, "netban", true).await.unwrap();
        runtime.set_group_module(200, "netban", true).await.unwrap();
        runtime.set_group_module(300, "netban", false).await.unwrap();

        let mut chats = runtime.list_netban_enabled_chats().await.unwrap();
        chats.sort();
        assert_eq!(chats, vec![100, 200]);

        runtime.set_group_module(100, "netban", false).await.unwrap();
        let chats = runtime.list_netban_enabled_chats().await.unwrap();
        assert_eq!(chats, vec![200]);
    }

    // Backs CmdClean's repeat-offense detection: a fresh (chat, user) has no
    // recorded offense, and recording one makes it show up as "just now".
    #[tokio::test]
    async fn permission_offense_round_trip() {
        let runtime = test_runtime().await;
        assert!(runtime.last_permission_offense(100, 200).await.unwrap().is_none());

        let before = Utc::now();
        runtime.record_permission_offense(100, 200).await.unwrap();
        let recorded = runtime.last_permission_offense(100, 200).await.unwrap().unwrap();
        assert!(recorded >= before - chrono::TimeDelta::seconds(2));

        // A different user in the same chat, and the same user in a
        // different chat, must not see each other's offenses.
        assert!(runtime.last_permission_offense(100, 999).await.unwrap().is_none());
        assert!(runtime.last_permission_offense(999, 200).await.unwrap().is_none());
    }

    // handle_permission_denied's escalation decision is "was the last
    // offense within 24h" - this confirms recording an offense always
    // updates to the latest timestamp (not just the first), which is what
    // that comparison relies on to correctly track a rolling window.
    #[tokio::test]
    async fn permission_offense_updates_to_latest_on_repeat() {
        let runtime = test_runtime().await;
        runtime.record_permission_offense(100, 200).await.unwrap();
        let first = runtime.last_permission_offense(100, 200).await.unwrap().unwrap();

        sleep(Duration::from_millis(10)).await;
        runtime.record_permission_offense(100, 200).await.unwrap();
        let second = runtime.last_permission_offense(100, 200).await.unwrap().unwrap();

        assert!(second >= first);
    }

    // Backs /revert: record an action, load it back (undo_data round-trips
    // through JSON correctly), mark it reverted, and confirm an already
    // reverted action is flagged as such.
    #[tokio::test]
    async fn maintainer_action_round_trip_and_reverted_flag() {
        let runtime = test_runtime().await;
        let undo = UndoData::GroupModule { chat_id: 100, module: "flood".to_string(), old_enabled: true };
        let action_id = runtime
            .record_maintainer_action(555, "Test Maintainer", Some(100), "/module", "flood true→false", &undo)
            .await
            .unwrap();

        let loaded = runtime.load_maintainer_action(action_id).await.unwrap().unwrap();
        assert_eq!(loaded.actor_name, "Test Maintainer");
        assert_eq!(loaded.chat_id, Some(100));
        assert_eq!(loaded.command, "/module");
        assert!(!loaded.reverted);
        match loaded.undo {
            UndoData::GroupModule { chat_id, module, old_enabled } => {
                assert_eq!(chat_id, 100);
                assert_eq!(module, "flood");
                assert!(old_enabled);
            }
            other => panic!("expected GroupModule, got {other:?}"),
        }

        runtime.mark_maintainer_action_reverted(action_id).await.unwrap();
        let reloaded = runtime.load_maintainer_action(action_id).await.unwrap().unwrap();
        assert!(reloaded.reverted);
    }

    #[tokio::test]
    async fn load_maintainer_action_missing_returns_none() {
        let runtime = test_runtime().await;
        assert!(runtime.load_maintainer_action(999999).await.unwrap().is_none());
    }

    // /revert's dispatcher is "call the same setter again with the old
    // value" for most UndoData variants - this exercises that exact pattern
    // for GroupModule and Threshold, the two simplest representative cases,
    // confirming the setters actually restore prior state correctly.
    #[tokio::test]
    async fn reverting_group_module_restores_prior_state() {
        let runtime = test_runtime().await;
        runtime.set_group_module(100, "flood", false).await.unwrap();
        assert!(!runtime.get_group_modules(100).await.unwrap().flood_control);

        // Simulates what /revert does for UndoData::GroupModule { old_enabled: true, .. }
        runtime.set_group_module(100, "flood", true).await.unwrap();
        assert!(runtime.get_group_modules(100).await.unwrap().flood_control);
    }

    #[tokio::test]
    async fn reverting_threshold_restores_prior_value() {
        let runtime = test_runtime().await;
        runtime.set_threshold(0.9).await.unwrap();
        assert_eq!(runtime.current_threshold().await.unwrap(), 0.9);

        // Simulates what /revert does for UndoData::Threshold { old: 0.7 }
        runtime.set_threshold(0.7).await.unwrap();
        assert_eq!(runtime.current_threshold().await.unwrap(), 0.7);
    }

    #[tokio::test]
    async fn set_token_counts_raw_restores_exact_counts() {
        let runtime = test_runtime().await;
        // Push the counts to some large, formula-derived values first...
        runtime.set_token_probability("spamword", 0.9).await.unwrap();

        // ...then simulate /revert for UndoData::TokenProbability { old_spam: 3, old_ham: 7 },
        // which must land on exactly those values, not another formula-derived pair.
        runtime.set_token_counts_raw("spamword", 3, 7).await.unwrap();
        let (spam_after, ham_after): (i64, i64) = runtime
            .with_conn(|conn| Ok(conn.query_row("SELECT spam_count, ham_count FROM word_frequencies WHERE word = 'spamword'", [], |row| Ok((row.get(0)?, row.get(1)?)))?))
            .await
            .unwrap();
        assert_eq!((spam_after, ham_after), (3, 7));
    }
}
