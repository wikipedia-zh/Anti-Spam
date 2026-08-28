// Spam Protection Bot (SPB)
// Copyright (c) 2026 Pepper. All rights reserved.
//
// Source-available, NOT open source. No license is granted. The source is
// published for transparency/audit and because Toolforge requires it; it
// grants no permission to use, modify, run, deploy, or host this software.
// Only the copyright holder and those they authorize may operate it. See
// the LICENSE file.

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
    data_dir: PathBuf,
    sqlite_path: PathBuf,
    spam_threshold: f64,
    // Optional: if set, a "bot is up" DM (with version/commit) is sent here
    // on every startup. Env-configured rather than hardcoded so a personal
    // Telegram user ID never ends up committed to source control.
    owner_id: Option<i64>,
    /// Second factor for the host console (see the HostCtl handler). A secret
    /// passphrase kept in an env var, never in source, so the host console
    /// needs both the host's authenticated account AND this secret. Guards
    /// against a compromised/left-open host account, and stays effective if
    /// the source ever goes closed. If unset, the console falls back to
    /// host-only with no second factor.
    hostctl_secret: Option<String>,
    /// The host console's trigger word, from env HOSTCTL_CMD (default
    /// "/hostctl"). Kept in an env var so the real command string isn't in
    /// source - lets the host pick something non-obvious to type in public
    /// groups. Not a security boundary on its own (is_host + hostctl_secret
    /// are), just a smaller footprint for shoulder-surfing.
    hostctl_cmd: String,
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
        let data_dir = env::var("DATA_DIR").unwrap_or_else(|_| "data".to_string());
        let sqlite_path = env::var("SQLITE_PATH").unwrap_or_else(|_| format!("{data_dir}/bot.db"));
        let spam_threshold = env::var("SPAM_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.85);
        let owner_id = env::var("OWNER_ID").ok().and_then(|v| v.parse::<i64>().ok());
        let hostctl_secret = env::var("HOSTCTL_SECRET").ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let hostctl_cmd = env::var("HOSTCTL_CMD").ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty()).unwrap_or_else(|| "/hostctl".to_string());
        Ok(Self {
            bot_token,
            log_channel_id,
            report_channel_id,
            test_group_id,
            data_dir: PathBuf::from(data_dir),
            sqlite_path: PathBuf::from(sqlite_path),
            spam_threshold,
            owner_id,
            hostctl_secret,
            hostctl_cmd,
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
    GuestBotBan,
    GuestInvokerBan,
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
            ActionKind::GuestBotBan => "guest_bot_ban",
            ActionKind::GuestInvokerBan => "guest_invoker_ban",
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
            "guest_bot_ban" => ActionKind::GuestBotBan,
            "guest_invoker_ban" => ActionKind::GuestInvokerBan,
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
    GroupModule { chat_id: i64, module: String, old_enabled: bool },
    /// `/module all on|off` - one audit entry covering every module it
    /// touched, so a single `/revert` puts all of them back rather than
    /// needing one per module.
    GroupModulesBulk { chat_id: i64, old: Vec<(String, bool)> },
    GroupWhitelist { chat_id: i64, user_id: i64, old_enabled: bool },
    GlobalWhitelist { user_id: i64, old_enabled: bool },
    RuleAdded { rule_id: i64 },
    RuleEdited { rule_id: i64, old_pattern: String },
    RuleDeleted { pattern: String, description: String },
    ProjectChat { old: Option<i64> },
    /// `/leave` and `/forbid` - reverting either just lifts the denial,
    /// same as `/forgive` would. Note `/leave` also made the bot leave the
    /// group; reverting can't undo that (the bot has to be re-invited), it
    /// only clears the blacklist so a re-invite is accepted.
    GroupBanned { chat_id: i64 },
    UserBanned { user_id: i64 },
    Reviewer { user_id: i64, old_enabled: bool },
    Maintainer { user_id: i64, old_enabled: bool },
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
    /// Private broadcast channel used as the SCP-079 exchange bus for the PM
    /// ticket bot's self-appeal bridge, set via `/set_exchange_channel`. Same
    /// persistence pattern as `project_chat`/`audit_log_chat`.
    exchange_channel: Mutex<Option<i64>>,
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
    /// chat_id -> last few human (non-bot) messages, newest last. Backs
    /// check_guest_bot_and_act's invoker correlation: Bot API never tells us
    /// who summoned a guest-mode bot (that link only exists in MTProto's
    /// `guestchat_via_from`, not the classic Bot API surface teloxide uses),
    /// so the only way to catch the human who typed `@thatbot` is to look at
    /// what was actually said just before the guest reply landed. In-memory
    /// only, small fixed cap per chat (see record_recent_message) - this is
    /// short-lived correlation context, not something worth persisting.
    recent_messages: Mutex<HashMap<i64, VecDeque<RecentMessage>>>,
    /// Project-level denial lists (see `migrate_v9_to_v10`), mirrored in
    /// memory because they're consulted on the hot path - every message and
    /// every command - where a SQLite round trip per check would be pure
    /// waste for something that changes a handful of times a year. The DB
    /// stays the source of truth; these are refilled from it at startup and
    /// written through on every change.
    banned_groups: RwLock<std::collections::HashSet<i64>>,
    banned_users: RwLock<std::collections::HashSet<i64>>,
    /// Command-granted maintainers (see migrate_v12_to_v13). Cached in
    /// memory like the other permission sets because it is read on every
    /// message (moderation exemption) and every maintainer command. The
    /// host is not stored here - it is a source constant - so this can be
    /// empty while the host still has full authority.
    maintainers: RwLock<std::collections::HashSet<i64>>,
    /// This bot's own account id, resolved once. `get_me()` was being called
    /// per message via `ensure_bot_can_moderate` (and again per bot-authored
    /// message via the guest-mode check) purely to learn an id that cannot
    /// change while the process runs. Every one of those was a network round
    /// trip against Telegram's rate limit before the message could even be
    /// looked at.
    me_id: OnceLock<UserId>,
}

/// One entry in `Runtime::recent_messages`. See that field's doc comment.
struct RecentMessage {
    user_id: i64,
    message_id: MessageId,
    display_name: String,
    text: String,
    seen_at: Instant,
}

struct PendingCaptcha {
    expected_answer: String,
    expires_at: Instant,
    challenge_message_id: MessageId,
}

#[derive(Clone)]
struct GroupModuleSettings {
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
    // Customized, non-public warning-escalates-to-ban module ("warn-pol").
    // Unlike every other module here, a group admin can't just turn this on
    // - it only takes effect if the chat is also on `module_allowlist`,
    // set via the maintainer-only `/magic` command. See ModerationCommand::Module.
    pol: bool,
    // Bans any message "from" a bot account that isn't actually a member of
    // the chat - the signature of Telegram's "guest mode" (any user can
    // @-mention a bot into posting directly into a group it was never added
    // to: https://core.telegram.org/api/bots/guest-mode). Defaults on, like
    // flood_control: baseline hygiene against an actively-exploited feature,
    // not an opinionated per-group choice, and it only ever fires on
    // messages from bots that were never legitimately added anyway.
    guest_ban: bool,
    // Attachment-policy modules (see migrate_v11_to_v12). Each bans on the
    // message carrying a particular kind of attachment, independent of any
    // text it has. All opt-in.
    no_contact: bool,
    no_voice: bool,
    no_exec: bool,
}

impl Default for GroupModuleSettings {
    fn default() -> Self {
        Self {
            no_halal: false,
            no_service_messages: false,
            flood_control: true,
            captcha: false,
            spam_threshold_override: None,
            netban: false,
            cmd_clean: false,
            pol: false,
            guest_ban: true,
            no_contact: false,
            no_voice: false,
            no_exec: false,
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
        let exchange_channel = Self::load_exchange_channel(&conn)?;
        let spam_rules = Self::load_spam_rules(&conn)?;
        let banned_groups = Self::load_id_set(&conn, "SELECT chat_id FROM banned_groups")?;
        let banned_users = Self::load_id_set(&conn, "SELECT user_id FROM banned_users")?;
        let maintainers = Self::load_id_set(&conn, "SELECT user_id FROM maintainers")?;
        Ok(Self {
            config,
            db: Arc::new(StdMutex::new(conn)),
            project_chat: Mutex::new(project_chat),
            audit_log_chat: Mutex::new(audit_log_chat),
            exchange_channel: Mutex::new(exchange_channel),
            model: Mutex::new(model),
            spam_rules: RwLock::new(spam_rules),
            mass_train_buffer: Mutex::new(HashMap::new()),
            mass_train_mode: Mutex::new(HashMap::new()),
            pending_rule_additions: Mutex::new(HashMap::new()),
            group_module_cache: RwLock::new(HashMap::new()),
            flood_tracker: Mutex::new(HashMap::new()),
            pending_captcha: Mutex::new(HashMap::new()),
            recent_messages: Mutex::new(HashMap::new()),
            banned_groups: RwLock::new(banned_groups),
            banned_users: RwLock::new(banned_users),
            maintainers: RwLock::new(maintainers),
            me_id: OnceLock::new(),
        })
    }

    fn load_id_set(conn: &Connection, sql: &str) -> Result<std::collections::HashSet<i64>> {
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map([], |row| row.get::<_, i64>(0))?;
        let mut out = std::collections::HashSet::new();
        for row in rows {
            out.insert(row?);
        }
        Ok(out)
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
        if user_version < 7 {
            Self::migrate_v6_to_v7(conn)?;
        }
        if user_version < 8 {
            Self::migrate_v7_to_v8(conn)?;
        }
        if user_version < 9 {
            Self::migrate_v8_to_v9(conn)?;
        }
        if user_version < 10 {
            Self::migrate_v9_to_v10(conn)?;
        }
        if user_version < 11 {
            Self::migrate_v10_to_v11(conn)?;
        }
        if user_version < 12 {
            Self::migrate_v11_to_v12(conn)?;
        }
        if user_version < 13 {
            Self::migrate_v12_to_v13(conn)?;
        }
        if user_version < 14 {
            Self::migrate_v13_to_v14(conn)?;
        }
        Ok(())
    }

    /// `ALTER TABLE ... ADD COLUMN`, skipped when the column is already
    /// there. SQLite has no `IF NOT EXISTS` for this, so a migration that
    /// ran against a schema already carrying the column would abort the
    /// whole transaction. Checking first makes each migration safe to
    /// re-run, which matters when one partially applied.
    fn add_column_if_missing(tx: &rusqlite::Transaction, table: &str, column: &str, decl: &str) -> Result<()> {
        let existing: i64 = tx.query_row(
            &format!("SELECT COUNT(*) FROM pragma_table_info('{table}') WHERE name = ?1"),
            params![column],
            |row| row.get(0),
        )?;
        if existing == 0 {
            tx.execute(&format!("ALTER TABLE {table} ADD COLUMN {column} {decl}"), [])?;
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
        Self::add_column_if_missing(&tx, "group_module_settings", "flood_control", "INTEGER NOT NULL DEFAULT 1")?;
        Self::add_column_if_missing(&tx, "group_module_settings", "captcha", "INTEGER NOT NULL DEFAULT 0")?;
        Self::add_column_if_missing(&tx, "group_module_settings", "spam_threshold_override", "REAL")?;
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
        Self::add_column_if_missing(&tx, "group_module_settings", "netban", "INTEGER NOT NULL DEFAULT 0")?;
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
        Self::add_column_if_missing(&tx, "group_module_settings", "cmd_clean", "INTEGER NOT NULL DEFAULT 0")?;
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

    /// Adds the `pol` opt-in flag (only settable once a chat is on
    /// `module_allowlist` - see `/magic`), `module_allowlist` itself (a
    /// generic maintainer-controlled per-(module, chat) gate, not specific
    /// to `pol` - a future customized module reuses this same table), and
    /// `pol_warnings`, a persistent per-(chat, user) count that - unlike
    /// `permission_offenses` - never expires: a second warning is always a
    /// ban, no matter how long ago the first one was.
    fn migrate_v6_to_v7(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        Self::add_column_if_missing(&tx, "group_module_settings", "pol", "INTEGER NOT NULL DEFAULT 0")?;
        tx.execute(
            r#"
            CREATE TABLE IF NOT EXISTS module_allowlist (
                module TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                added_by INTEGER,
                created_at TEXT NOT NULL,
                PRIMARY KEY (module, chat_id)
            )
            "#,
            [],
        )?;
        tx.execute(
            r#"
            CREATE TABLE IF NOT EXISTS pol_warnings (
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                warn_count INTEGER NOT NULL DEFAULT 0,
                last_warned_at TEXT NOT NULL,
                PRIMARY KEY (chat_id, user_id)
            )
            "#,
            [],
        )?;
        tx.execute("PRAGMA user_version = 7", [])?;
        tx.commit()?;
        Ok(())
    }

    /// Adds the `guest_ban` opt-in flag (defaults on - see the field's doc
    /// comment on `GroupModuleSettings`) backing the new guest-mode-bot
    /// auto-ban check.
    fn migrate_v7_to_v8(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        Self::add_column_if_missing(&tx, "group_module_settings", "guest_ban", "INTEGER NOT NULL DEFAULT 1")?;
        tx.execute("PRAGMA user_version = 8", [])?;
        tx.commit()?;
        Ok(())
    }

    /// Makes netban membership an explicit, immutable property of the case
    /// rather than something re-derived from the origin group's *current*
    /// settings on every lookup. Two reasons that mattered:
    ///
    /// 1. A group with a lowered `spam_threshold_override` was minting
    ///    network bans at its own bar. Set it low enough and every message
    ///    in that group becomes a project-wide ban - so the shared
    ///    blacklist was only ever as trustworthy as the least strict group
    ///    on it. Eligibility is now judged against the global threshold.
    /// 2. Deriving from live settings meant a group toggling netban off
    ///    silently retracted every ban it had ever contributed, and
    ///    toggling it on retroactively promoted its whole ban history.
    ///
    /// Backfill applies `netban_eligible`'s rule to the existing history
    /// rather than preserving whatever the old JOIN happened to return.
    /// That deliberately *drops* past entries the new rule rejects - `/sb`
    /// bans and unscored module bans that only qualified because their group
    /// was on the network. Those are exactly the group-level decisions that
    /// shouldn't have been project-wide, so leaving them in place would keep
    /// enforcing the thing this change exists to stop.
    fn migrate_v8_to_v9(conn: &mut Connection) -> Result<()> {
        let threshold = Self::load_threshold(conn)?.unwrap_or(0.85);
        let tx = conn.transaction()?;
        Self::add_column_if_missing(&tx, "cases", "netban_eligible", "INTEGER NOT NULL DEFAULT 0")?;
        tx.execute(
            r#"
            UPDATE cases SET netban_eligible = 1
            WHERE action IN ('report_approved', 'guest_bot_ban', 'guest_invoker_ban')
               OR (action = 'auto_ban' AND model_score IS NOT NULL AND model_score >= ?1)
            "#,
            params![threshold],
        )?;
        tx.execute("PRAGMA user_version = 9", [])?;
        tx.commit()?;
        Ok(())
    }

    /// Project-level denial lists, distinct from every other ban in this
    /// file: those are moderation *inside* a group, these are "this group /
    /// this person may not use the project at all". `/leave` writes to
    /// `banned_groups`, `/forbid` to `banned_users`, and `/forgive` clears
    /// either. Separate from `group_whitelist`/`global_whitelist` (which
    /// exempt people from moderation) and from `cases` (per-incident
    /// records) - this is standing access control, with no case attached.
    fn migrate_v9_to_v10(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute(
            r#"
            CREATE TABLE IF NOT EXISTS banned_groups (
                chat_id INTEGER PRIMARY KEY,
                reason TEXT NOT NULL DEFAULT '',
                added_by INTEGER,
                created_at TEXT NOT NULL
            )
            "#,
            [],
        )?;
        tx.execute(
            r#"
            CREATE TABLE IF NOT EXISTS banned_users (
                user_id INTEGER PRIMARY KEY,
                reason TEXT NOT NULL DEFAULT '',
                added_by INTEGER,
                created_at TEXT NOT NULL
            )
            "#,
            [],
        )?;
        tx.execute("PRAGMA user_version = 10", [])?;
        tx.commit()?;
        Ok(())
    }

    /// The reviewer role: may act on the approve/reject buttons in the
    /// report channel, and nothing else. Until now those buttons were
    /// guarded only by the message being *in* the report channel, so
    /// anyone who could see the channel could approve a ban or push text
    /// into the shared model. This makes that an explicit grant.
    ///
    /// Deliberately not a `maintainer_ids`-style env var: reviewers are
    /// expected to change far more often than maintainers, and rotating one
    /// shouldn't need a redeploy.
    fn migrate_v10_to_v11(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute(
            r#"
            CREATE TABLE IF NOT EXISTS reviewers (
                user_id INTEGER PRIMARY KEY,
                added_by INTEGER,
                created_at TEXT NOT NULL
            )
            "#,
            [],
        )?;
        tx.execute("PRAGMA user_version = 11", [])?;
        tx.commit()?;
        Ok(())
    }

    /// Three attachment-policy modules, and the counter behind `/spam`'s
    /// three-strike rule.
    ///
    /// The modules ban on message *type* rather than content: a shared
    /// contact card, a voice recording, or an executable attachment. All
    /// default off, like every other opinionated policy module - a group
    /// that legitimately shares recordings or installers should not have
    /// its members banned because the bot was upgraded.
    ///
    /// `report_offenses` counts rejected `/spam` reports per reporter.
    /// It can't be derived from `cases`: rejecting a report overwrites the
    /// case's `actor_user_id` with the reviewer's id, so the reporter is no
    /// longer recorded there by the time the rejection is final.
    fn migrate_v11_to_v12(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        Self::add_column_if_missing(&tx, "group_module_settings", "no_contact", "INTEGER NOT NULL DEFAULT 0")?;
        Self::add_column_if_missing(&tx, "group_module_settings", "no_voice", "INTEGER NOT NULL DEFAULT 0")?;
        Self::add_column_if_missing(&tx, "group_module_settings", "no_exec", "INTEGER NOT NULL DEFAULT 0")?;
        tx.execute(
            r#"
            CREATE TABLE IF NOT EXISTS report_offenses (
                user_id INTEGER PRIMARY KEY,
                rejected_count INTEGER NOT NULL DEFAULT 0,
                last_rejected_at TEXT NOT NULL
            )
            "#,
            [],
        )?;
        tx.execute("PRAGMA user_version = 12", [])?;
        tx.commit()?;
        Ok(())
    }

    /// The maintainer role, now granted by command and stored here rather
    /// than read from a `MAINTAINER_IDS` env var. The env var no longer
    /// confers any authority - which is exactly the "remove all current
    /// maintainers" step: on first run against an existing database this
    /// table is empty, so nobody but the hardcoded host (see `HOST_ID`) has
    /// power until the host grants it.
    fn migrate_v12_to_v13(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute(
            r#"
            CREATE TABLE IF NOT EXISTS maintainers (
                user_id INTEGER PRIMARY KEY,
                added_by INTEGER,
                created_at TEXT NOT NULL
            )
            "#,
            [],
        )?;
        tx.execute("PRAGMA user_version = 13", [])?;
        tx.commit()?;
        Ok(())
    }

    /// Remembers where a `/spam` reporter's confirmation message lives, so
    /// its text can be updated once the report is decided in the review
    /// channel. Persisted rather than kept in memory because a review can
    /// happen hours later, or after a redeploy - an in-memory map would lose
    /// the message id across the very restarts that happen most often.
    fn migrate_v13_to_v14(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute(
            r#"
            CREATE TABLE IF NOT EXISTS report_confirmations (
                case_id TEXT PRIMARY KEY,
                chat_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL
            )
            "#,
            [],
        )?;
        tx.execute("PRAGMA user_version = 14", [])?;
        tx.commit()?;
        Ok(())
    }

    /// A `/set` call with a target very close to 0 or 1 used to compute an
    /// astronomically large raw count for a single token (see
    /// the removed `/set` command) — that count then dominated the shared
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
        // spam_docs/ham_docs are derived from training_samples directly, not
        // read back from the model_meta counters those same rows are
        // supposed to keep in sync - a maintenance pass (bulk import, a
        // manual data-cleanup script) that touched training_samples without
        // going through train_spam/train_ham's increment would otherwise
        // leave model_meta silently drifted from reality forever. Deriving
        // it fresh here makes every restart (and every rebuild_model call
        // below) self-healing regardless of how the drift happened.
        let spam_docs = conn.query_row("SELECT COUNT(*) FROM training_samples WHERE label = 'spam'", [], |row| row.get::<_, i64>(0))? as u64;
        let ham_docs = conn.query_row("SELECT COUNT(*) FROM training_samples WHERE label = 'ham'", [], |row| row.get::<_, i64>(0))? as u64;
        let mut model = ModelState { spam_docs, ham_docs, ..Default::default() };

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

    fn load_exchange_channel(conn: &Connection) -> Result<Option<i64>> {
        let mut stmt = conn.prepare("SELECT value FROM model_meta WHERE key = 'exchange_channel_id'")?;
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

    /// Finds the most recent active ban for `user_id` that made it onto the
    /// shared blacklist - i.e. "is this user currently network-banned".
    /// Keyed off the case's own `netban_eligible` flag (set once, at ban
    /// time, by `propagate_network_ban`) rather than the origin group's
    /// live settings, so a group can't retract or promote its entire ban
    /// history just by toggling its own netban switch - and so a group
    /// running a lowered threshold can't mint entries below the project
    /// bar. See `migrate_v8_to_v9`.
    ///
    /// Reuses the same "reversal mutates action in place" property as
    /// `load_latest_case_by_actions`: once reversed, a case's action becomes
    /// `Unbanned` and stops matching the `IN (...)` filter here too, so no
    /// separate "is this stale" bookkeeping is needed.
    async fn find_active_network_ban(&self, user_id: i64) -> Result<Option<CaseRecord>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                // The action filter is only here to drop reversed cases -
                // /unban rewrites `action` to 'unbanned' in place, so a
                // lifted ban stops matching. Guest-mode bans are listed
                // because they're netban-eligible (see `netban_eligible`);
                // without them here a guest bot would be propagated at ban
                // time but forgotten by the time it joined somewhere new.
                "SELECT id, action, chat_id, target_user_id, target_name, actor_user_id, actor_name, source_message_id, evidence_text, model_score, matched_rule_id, matched_rule_pattern, status, log_message_id, created_at
                 FROM cases
                 WHERE target_user_id = ?1 AND netban_eligible = 1
                   AND action IN ('auto_ban', 'spam_ban', 'report_approved', 'guest_bot_ban', 'guest_invoker_ban')
                 ORDER BY created_at DESC LIMIT 1",
            )?;
            let mut rows = stmt.query(params![user_id])?;
            rows.next()?.map(case_from_row).transpose()
        })
        .await
    }

    /// Adds a case to the shared blacklist. Separate from `persist_case` on
    /// purpose: that one rewrites the whole row from a `CaseRecord`, which
    /// carries no netban field, so leaving this column out of its INSERT and
    /// its ON CONFLICT list means a later re-persist (a status change, a log
    /// id backfill) can never silently clear the flag.
    async fn mark_netban_eligible(&self, case_id: &str) -> Result<()> {
        let case_id = case_id.to_string();
        self.with_conn(move |conn| {
            conn.execute("UPDATE cases SET netban_eligible = 1 WHERE id = ?1", params![case_id])?;
            Ok(())
        })
        .await
    }

    /// Finds the most recent active ban for `user_id` in this *exact* chat,
    /// with no netban involvement at all - that flag only governs whether a
    /// ban propagates to *other* chats, and is irrelevant to "did this
    /// specific chat already ban this person". Backs check_reban_and_act's
    /// same-chat ban-evasion safety net.
    async fn find_active_ban_in_chat(&self, chat_id: i64, user_id: i64) -> Result<Option<CaseRecord>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, action, chat_id, target_user_id, target_name, actor_user_id, actor_name, source_message_id, evidence_text, model_score, matched_rule_id, matched_rule_pattern, status, log_message_id, created_at
                 FROM cases WHERE chat_id = ?1 AND target_user_id = ?2 AND action IN ('auto_ban', 'spam_ban', 'report_approved')
                 ORDER BY created_at DESC LIMIT 1",
            )?;
            let mut rows = stmt.query(params![chat_id, user_id])?;
            rows.next()?.map(case_from_row).transpose()
        })
        .await
    }

    /// Same "reversal mutates action in place" property as
    /// `find_active_network_ban`, but without its netban-membership join -
    /// this is for the PM appeal bridge, which needs to see a ban in *any*
    /// group, not just ones that opted into netban propagation. A user can
    /// have independent active bans in several unrelated groups at once, so
    /// this returns all of them rather than just the most recent.
    async fn find_active_bans_for_user(&self, user_id: i64) -> Result<Vec<CaseRecord>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, action, chat_id, target_user_id, target_name, actor_user_id, actor_name, source_message_id, evidence_text, model_score, matched_rule_id, matched_rule_pattern, status, log_message_id, created_at
                 FROM cases WHERE target_user_id = ?1 AND action IN ('auto_ban', 'spam_ban', 'report_approved', 'guest_bot_ban', 'guest_invoker_ban')
                 ORDER BY created_at DESC",
            )?;
            let mut rows = stmt.query(params![user_id])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push(case_from_row(row)?);
            }
            Ok(out)
        })
        .await
    }

    /// "How many times has this person been banned" for the PM bridge's
    /// `strike_count` field - counts bans that were later lifted too
    /// (`unbanned`), since a strike is about history, not current status.
    async fn count_ban_strikes_for_user(&self, user_id: i64) -> Result<i64> {
        self.with_conn(move |conn| {
            Ok(conn.query_row(
                "SELECT COUNT(*) FROM cases WHERE target_user_id = ?1 AND action IN ('auto_ban', 'spam_ban', 'report_approved', 'unbanned')",
                params![user_id],
                |row| row.get(0),
            )?)
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

    /// Maintenance pass over `training_samples`, idempotent (safe to
    /// re-run): drops empty/whitespace-only samples entirely (zero token
    /// signal even singly, just pollutes the doc-count denominator), and
    /// for every exact-duplicate (label, text) group beyond the first
    /// occurrence, rolls back its word_frequencies/doc-count contribution
    /// and deletes the row - same accounting as `purge_training_by_case`,
    /// just applied per-duplicate-row instead of per-case_id (many mass-
    /// imported duplicates never had a real case_id to key off of at all).
    /// Returns (duplicates_removed, empty_removed). Caller must call
    /// `rebuild_model()` afterward.
    async fn dedupe_training_samples(&self) -> Result<(usize, usize)> {
        self.with_conn(move |conn| {
            let tx = conn.transaction()?;
            let mut spam_docs: i64 = tx.query_row("SELECT COALESCE(value, '0') FROM model_meta WHERE key = 'spam_docs'", [], |row| row.get::<_, String>(0))?.parse().unwrap_or(0);
            let mut ham_docs: i64 = tx.query_row("SELECT COALESCE(value, '0') FROM model_meta WHERE key = 'ham_docs'", [], |row| row.get::<_, String>(0))?.parse().unwrap_or(0);

            fn rollback(tx: &rusqlite::Transaction, text: &str, label: &str) -> rusqlite::Result<()> {
                for token in tokenize(text) {
                    let counts = tx.query_row(
                        "SELECT spam_count, ham_count FROM word_frequencies WHERE word = ?1",
                        params![&token],
                        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
                    );
                    if let Ok((mut spam_count, mut ham_count)) = counts {
                        match label {
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
                Ok(())
            }

            // Empty/whitespace-only samples: remove every copy, no signal even singly.
            let mut empty_removed = 0usize;
            {
                let to_delete: Vec<(i64, String)> = {
                    let mut stmt = tx.prepare("SELECT rowid, label FROM training_samples WHERE trim(text) = ''")?;
                    let mut rows = stmt.query([])?;
                    let mut out = Vec::new();
                    while let Some(row) = rows.next()? {
                        out.push((row.get::<_, i64>(0)?, row.get::<_, String>(1)?));
                    }
                    out
                };
                for (rowid, label) in to_delete {
                    match label.as_str() {
                        "spam" => spam_docs = (spam_docs - 1).max(0),
                        "ham" => ham_docs = (ham_docs - 1).max(0),
                        _ => {}
                    }
                    tx.execute("DELETE FROM training_samples WHERE rowid = ?1", params![rowid])?;
                    empty_removed += 1;
                }
            }

            // Exact-duplicate (label, text) groups: keep the earliest row, purge the rest.
            let mut dup_removed = 0usize;
            {
                let groups: Vec<(String, String, i64)> = {
                    let mut stmt = tx.prepare(
                        "SELECT label, text, MIN(rowid) FROM training_samples WHERE trim(text) != '' GROUP BY label, text HAVING COUNT(*) > 1",
                    )?;
                    let mut rows = stmt.query([])?;
                    let mut out = Vec::new();
                    while let Some(row) = rows.next()? {
                        out.push((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, i64>(2)?));
                    }
                    out
                };
                for (label, text, keep_rowid) in groups {
                    let extra_rowids: Vec<i64> = {
                        let mut stmt = tx.prepare("SELECT rowid FROM training_samples WHERE label = ?1 AND text = ?2 AND rowid != ?3")?;
                        let mut rows = stmt.query(params![&label, &text, keep_rowid])?;
                        let mut out = Vec::new();
                        while let Some(row) = rows.next()? {
                            out.push(row.get::<_, i64>(0)?);
                        }
                        out
                    };
                    for rowid in extra_rowids {
                        rollback(&tx, &text, &label)?;
                        match label.as_str() {
                            "spam" => spam_docs = (spam_docs - 1).max(0),
                            "ham" => ham_docs = (ham_docs - 1).max(0),
                            _ => {}
                        }
                        tx.execute("DELETE FROM training_samples WHERE rowid = ?1", params![rowid])?;
                        dup_removed += 1;
                    }
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

            tx.commit()?;
            Ok((dup_removed, empty_removed))
        })
        .await
    }

    /// Hold-out evaluation: trains a throwaway model on part of the samples
    /// and scores the rest, which the model has never seen.
    ///
    /// Exists because the spam threshold was being chosen by feel. Scoring
    /// the live model against its own training data would just report how
    /// well it memorised them, so this splits the samples deterministically
    /// (every Nth row by rowid, no RNG - the same data gives the same answer
    /// twice) and builds the counts in memory. Nothing here touches
    /// `word_frequencies`, so it is safe to run on a live bot.
    ///
    /// Reports precision/recall at several thresholds, since the useful
    /// question is not "is 0.85 good" but "what does moving it cost".
    async fn evaluate_model(&self, holdout_fraction: f64) -> Result<String> {
        let samples: Vec<(i64, String, String)> = self
            .with_conn(|conn| {
                let mut stmt = conn.prepare("SELECT rowid, label, text FROM training_samples WHERE trim(text) != '' AND label IN ('spam','ham') ORDER BY rowid")?;
                let mut rows = stmt.query([])?;
                let mut out = Vec::new();
                while let Some(row) = rows.next()? {
                    out.push((row.get(0)?, row.get(1)?, row.get(2)?));
                }
                Ok(out)
            })
            .await?;

        if samples.len() < 20 {
            return Ok(format!("樣本太少（{}），至少需要 20 筆才能評估。", samples.len()));
        }

        let step = (1.0 / holdout_fraction).round().max(2.0) as usize;
        let mut train = ModelState::default();
        let mut test: Vec<(bool, String)> = Vec::new();
        for (i, (_, label, text)) in samples.iter().enumerate() {
            let is_spam = label == "spam";
            if i % step == 0 {
                test.push((is_spam, text.clone()));
                continue;
            }
            let tokens = tokenize(text);
            if is_spam {
                train.spam_docs += 1;
                for t in tokens { *train.spam_tokens.entry(t).or_default() += 1; }
            } else {
                train.ham_docs += 1;
                for t in tokens { *train.ham_tokens.entry(t).or_default() += 1; }
            }
        }
        if test.is_empty() || train.spam_docs == 0 || train.ham_docs == 0 {
            return Ok("切分後其中一側沒有樣本，無法評估。".to_string());
        }

        let scored: Vec<(bool, f64)> = test.iter().map(|(is_spam, text)| (*is_spam, score_spam_from_text(&train, text))).collect();
        let live = self.current_threshold().await.unwrap_or(self.config.spam_threshold);

        let mut out = format!(
            "<b>❖ 模型評估</b>\n訓練 {} 筆 / 測試 {} 筆（保留 1/{step}）\n目前門檻 {live:.2}\n\n<code>門檻  精確率  召回率   F1   漏放  誤封</code>",
            train.spam_docs + train.ham_docs,
            scored.len(),
        );
        let mut thresholds = vec![0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95];
        if !thresholds.iter().any(|t| (*t - live).abs() < 1e-9) {
            thresholds.push(live);
            thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        for t in thresholds {
            let (mut tp, mut fp, mut fnn) = (0usize, 0usize, 0usize);
            for (is_spam, score) in &scored {
                match (*is_spam, *score >= t) {
                    (true, true) => tp += 1,
                    (false, true) => fp += 1,
                    (true, false) => fnn += 1,
                    (false, false) => {}
                }
            }
            let precision = if tp + fp == 0 { 0.0 } else { tp as f64 / (tp + fp) as f64 };
            let recall = if tp + fnn == 0 { 0.0 } else { tp as f64 / (tp + fnn) as f64 };
            let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
            let marker = if (t - live).abs() < 1e-9 { " ←" } else { "" };
            out.push_str(&format!("\n<code>{t:.2}   {:.3}   {:.3}  {:.3}   {fnn:>3}   {fp:>3}</code>{marker}", precision, recall, f1));
        }
        out.push_str("\n\n漏放 = 是垃圾但沒攔到；誤封 = 正常訊息被判垃圾。\n此評估只用訓練樣本，未動到實際模型。");
        Ok(out)
    }

    /// Rebuilds `word_frequencies` from scratch by replaying every
    /// `training_samples` row through the *current* tokenizer.
    ///
    /// `rebuild_model` only re-reads the counts that are already stored, so
    /// it cannot repair them. Anything that changes how text becomes tokens
    /// (such as `collapse_origin_markers`) leaves the old counts in place,
    /// still carrying whatever the previous tokenizer produced. This is the
    /// only way to make such a change retroactive.
    ///
    /// Note this discards manual `/set` biases, which live in
    /// `word_frequencies` and have no backing sample to replay.
    async fn retrain_from_samples(&self) -> Result<(usize, usize)> {
        self.with_conn(|conn| {
            let tx = conn.transaction()?;
            let samples: Vec<(String, String)> = {
                let mut stmt = tx.prepare("SELECT label, text FROM training_samples WHERE trim(text) != ''")?;
                let mut rows = stmt.query([])?;
                let mut out = Vec::new();
                while let Some(row) = rows.next()? {
                    out.push((row.get(0)?, row.get(1)?));
                }
                out
            };

            tx.execute("DELETE FROM word_frequencies", [])?;
            let (mut spam_docs, mut ham_docs) = (0usize, 0usize);
            for (label, text) in &samples {
                let (spam_delta, ham_delta) = match label.as_str() {
                    "spam" => (1, 0),
                    "ham" => (0, 1),
                    _ => continue,
                };
                for token in tokenize(text) {
                    tx.execute(
                        "INSERT INTO word_frequencies (word, spam_count, ham_count) VALUES (?1, ?2, ?3)
                         ON CONFLICT(word) DO UPDATE SET spam_count = spam_count + ?2, ham_count = ham_count + ?3",
                        params![token, spam_delta, ham_delta],
                    )?;
                }
                if spam_delta == 1 { spam_docs += 1 } else { ham_docs += 1 }
            }

            tx.execute(
                "INSERT INTO model_meta (key, value) VALUES ('spam_docs', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![spam_docs.to_string()],
            )?;
            tx.execute(
                "INSERT INTO model_meta (key, value) VALUES ('ham_docs', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![ham_docs.to_string()],
            )?;
            tx.commit()?;
            Ok((spam_docs, ham_docs))
        })
        .await
    }

    /// Refreshes the in-memory model from disk. This only reads — the DB is
    /// always the source of truth and callers that changed the DB (train_spam,
    /// purge, undo, retrain, ...) have already persisted their
    /// specific changes, so there is nothing to write back here.
    async fn rebuild_model(&self) -> Result<ModelState> {
        let rebuilt = self
            .with_conn(|conn| {
                // See load_model's comment: derived from training_samples,
                // not the separately-tracked model_meta counters, so this
                // self-heals any drift between the two instead of persisting it.
                let spam_docs = conn.query_row("SELECT COUNT(*) FROM training_samples WHERE label = 'spam'", [], |row| row.get::<_, i64>(0))? as u64;
                let ham_docs = conn.query_row("SELECT COUNT(*) FROM training_samples WHERE label = 'ham'", [], |row| row.get::<_, i64>(0))? as u64;
                let mut rebuilt = ModelState { spam_docs, ham_docs, ..Default::default() };
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

    async fn current_threshold(&self) -> Result<f64> {
        let value = self.with_conn(|conn| Self::load_threshold(conn)).await?;
        Ok(value.unwrap_or(self.config.spam_threshold))
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

    async fn set_exchange_channel(&self, chat_id: i64) {
        let _ = self
            .with_conn(move |conn| {
                conn.execute(
                    "INSERT INTO model_meta (key, value) VALUES ('exchange_channel_id', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    params![chat_id.to_string()],
                )?;
                Ok(())
            })
            .await;
        let mut exchange_channel = self.exchange_channel.lock().await;
        *exchange_channel = Some(chat_id);
    }

    async fn exchange_channel(&self) -> Option<i64> {
        let exchange_channel = self.exchange_channel.lock().await;
        *exchange_channel
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

    /// The host-console secret. Stored in the database (model_meta) so it can
    /// be changed by command and recovered with a plain SQL query if
    /// forgotten. Falls back to the HOSTCTL_SECRET env var only when nothing
    /// is stored yet (first run).
    async fn hostctl_secret(&self) -> Option<String> {
        let stored = self
            .with_conn(|conn| {
                Ok(conn
                    .query_row("SELECT value FROM model_meta WHERE key = 'hostctl_secret'", [], |r| r.get::<_, String>(0))
                    .ok())
            })
            .await
            .ok()
            .flatten()
            .filter(|v| !v.is_empty());
        stored.or_else(|| self.config.hostctl_secret.clone())
    }

    async fn set_hostctl_secret(&self, secret: &str) -> Result<()> {
        let secret = secret.to_string();
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO model_meta (key, value) VALUES ('hostctl_secret', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![secret],
            )?;
            Ok(())
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

    async fn spam_rule_pattern_exists(&self, pattern: &str) -> bool {
        let pattern = pattern.to_string();
        self.with_conn(move |conn| {
            Ok(conn
                .query_row("SELECT COUNT(*) FROM spam_rules WHERE pattern = ?1", params![pattern], |row| row.get::<_, i64>(0))
                .unwrap_or(0)
                > 0)
        })
        .await
        .unwrap_or(false)
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
                // no_long_name column intentionally not selected: the module
                // was removed after too many false positives. The DB column
                // is left in place (harmless) rather than migrated away.
                let mut stmt = conn.prepare("SELECT no_halal, no_service_messages, flood_control, captcha, spam_threshold_override, netban, cmd_clean, pol, guest_ban, no_contact, no_voice, no_exec FROM group_module_settings WHERE chat_id = ?1")?;
                let mut rows = stmt.query(params![chat_id])?;
                if let Some(row) = rows.next()? {
                    Ok(GroupModuleSettings {
                        no_halal: row.get::<_, i64>(0)? != 0,
                        no_service_messages: row.get::<_, i64>(1)? != 0,
                        flood_control: row.get::<_, i64>(2)? != 0,
                        captcha: row.get::<_, i64>(3)? != 0,
                        spam_threshold_override: row.get::<_, Option<f64>>(4)?,
                        netban: row.get::<_, i64>(5)? != 0,
                        cmd_clean: row.get::<_, i64>(6)? != 0,
                        pol: row.get::<_, i64>(7)? != 0,
                        guest_ban: row.get::<_, i64>(8)? != 0,
                        no_contact: row.get::<_, i64>(9)? != 0,
                        no_voice: row.get::<_, i64>(10)? != 0,
                        no_exec: row.get::<_, i64>(11)? != 0,
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
            "warn-pol" => {
                conn.execute(
                    "UPDATE group_module_settings SET pol = ?2 WHERE chat_id = ?1",
                    params![chat_id, if enabled { 1 } else { 0 }],
                )?;
            }
            "guestban" => {
                conn.execute(
                    "UPDATE group_module_settings SET guest_ban = ?2 WHERE chat_id = ?1",
                    params![chat_id, if enabled { 1 } else { 0 }],
                )?;
            }
            "nocontact" => {
                conn.execute(
                    "UPDATE group_module_settings SET no_contact = ?2 WHERE chat_id = ?1",
                    params![chat_id, if enabled { 1 } else { 0 }],
                )?;
            }
            "novoice" => {
                conn.execute(
                    "UPDATE group_module_settings SET no_voice = ?2 WHERE chat_id = ?1",
                    params![chat_id, if enabled { 1 } else { 0 }],
                )?;
            }
            "noexec" => {
                conn.execute(
                    "UPDATE group_module_settings SET no_exec = ?2 WHERE chat_id = ?1",
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
        let tripped = timestamps.len() >= LIMIT;
        if timestamps.is_empty() {
            // The deque was trimmed to nothing, so this pair is idle. Drop the
            // key too - otherwise the map keeps one entry per (chat, user)
            // ever seen, for the life of the process.
            tracker.remove(&(chat_id, user_id));
        }
        tripped
    }

    /// Records a human message for check_guest_bot_and_act's invoker
    /// correlation (see `recent_messages`'s doc comment for why this exists
    /// at all). Capped at a handful of entries per chat - this only ever
    /// needs to answer "what did someone say a few seconds ago", not build
    /// any real history.
    async fn record_recent_message(&self, chat_id: i64, user_id: i64, message_id: MessageId, display_name: &str, text: &str) {
        const CAP: usize = 8;
        let mut buffers = self.recent_messages.lock().await;
        let entries = buffers.entry(chat_id).or_default();
        entries.push_back(RecentMessage {
            user_id,
            message_id,
            display_name: display_name.to_string(),
            text: text.to_string(),
            seen_at: Instant::now(),
        });
        while entries.len() > CAP {
            entries.pop_front();
        }
    }

    /// Scans this chat's recent human messages (most recent first) for one
    /// that bare-mentions `bot_username` - the literal way guest mode is
    /// invoked - within the last `WINDOW` of real time. "Bare" means what's
    /// left after stripping *every* @mention is short: real conversation
    /// that happens to reference a bot by name reads nothing like a
    /// guest-mode summon, which per Telegram's own docs is just the
    /// @-mention. Stripping all mentions rather than only this bot's is
    /// deliberate - a reported evasion summons several guest bots in one
    /// message (plus an emoji), which left the *other* bots' handles in the
    /// remainder and made every one of them look like ordinary chatter.
    /// This is a heuristic, not a certainty - Bot API has no field linking
    /// a guest reply back to whoever triggered it - so callers should treat
    /// a match as strong evidence, not proof.
    async fn find_recent_guest_invoker(&self, chat_id: i64, bot_username: &str) -> Option<(i64, MessageId, String, String)> {
        const WINDOW: std::time::Duration = std::time::Duration::from_secs(30);
        const BARE_MAX_REMAINDER: usize = 8;
        let mention = format!("@{}", bot_username.to_lowercase());
        let now = Instant::now();
        let buffers = self.recent_messages.lock().await;
        let entries = buffers.get(&chat_id)?;
        entries.iter().rev().find_map(|entry| {
            if now.duration_since(entry.seen_at) > WINDOW {
                return None;
            }
            let lower = entry.text.to_lowercase();
            if !lower.contains(&mention) {
                return None;
            }
            if strip_mentions(&lower).trim().chars().count() > BARE_MAX_REMAINDER {
                return None;
            }
            Some((entry.user_id, entry.message_id, entry.display_name.clone(), entry.text.clone()))
        })
    }

    /// Drops a specific recorded message, so a summon that pulled in several
    /// guest bots at once only ever bans its sender for the first reply to
    /// land - the rest find nothing and skip straight past.
    async fn forget_recent_message(&self, chat_id: i64, message_id: MessageId) {
        let mut buffers = self.recent_messages.lock().await;
        if let Some(entries) = buffers.get_mut(&chat_id) {
            entries.retain(|entry| entry.message_id != message_id);
        }
    }

    /// The bot's own id, fetched from Telegram once and then reused. Falls
    /// back to a live call only if the first attempt failed.
    async fn me_id(&self, bot: &Bot) -> Option<UserId> {
        if let Some(id) = self.me_id.get() {
            return Some(*id);
        }
        let id = bot.get_me().await.ok()?.id;
        let _ = self.me_id.set(id);
        Some(id)
    }

    /// Project-level denial checks. In-memory (see the `banned_groups` /
    /// `banned_users` fields), so these are cheap enough to call on every
    /// message without a DB round trip.
    async fn is_group_banned(&self, chat_id: i64) -> bool {
        self.banned_groups.read().await.contains(&chat_id)
    }

    async fn is_user_banned(&self, user_id: i64) -> bool {
        self.banned_users.read().await.contains(&user_id)
    }

    async fn set_group_banned(&self, chat_id: i64, banned: bool, reason: &str, added_by: Option<i64>) -> Result<()> {
        let reason = reason.to_string();
        self.with_conn(move |conn| {
            if banned {
                conn.execute(
                    "INSERT INTO banned_groups (chat_id, reason, added_by, created_at) VALUES (?1, ?2, ?3, ?4)
                     ON CONFLICT(chat_id) DO UPDATE SET reason=excluded.reason, added_by=excluded.added_by, created_at=excluded.created_at",
                    params![chat_id, reason, added_by, Utc::now().to_rfc3339()],
                )?;
            } else {
                conn.execute("DELETE FROM banned_groups WHERE chat_id = ?1", params![chat_id])?;
            }
            Ok(())
        })
        .await?;
        let mut cache = self.banned_groups.write().await;
        if banned {
            cache.insert(chat_id);
        } else {
            cache.remove(&chat_id);
        }
        Ok(())
    }

    async fn set_user_banned(&self, user_id: i64, banned: bool, reason: &str, added_by: Option<i64>) -> Result<()> {
        let reason = reason.to_string();
        self.with_conn(move |conn| {
            if banned {
                conn.execute(
                    "INSERT INTO banned_users (user_id, reason, added_by, created_at) VALUES (?1, ?2, ?3, ?4)
                     ON CONFLICT(user_id) DO UPDATE SET reason=excluded.reason, added_by=excluded.added_by, created_at=excluded.created_at",
                    params![user_id, reason, added_by, Utc::now().to_rfc3339()],
                )?;
            } else {
                conn.execute("DELETE FROM banned_users WHERE user_id = ?1", params![user_id])?;
            }
            Ok(())
        })
        .await?;
        let mut cache = self.banned_users.write().await;
        if banned {
            cache.insert(user_id);
        } else {
            cache.remove(&user_id);
        }
        Ok(())
    }

    /// Records the reporter-confirmation message for a case, so
    /// `take_report_confirmation` can find it when the report is decided.
    async fn set_report_confirmation(&self, case_id: &str, chat_id: i64, message_id: i32) -> Result<()> {
        let case_id = case_id.to_string();
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO report_confirmations (case_id, chat_id, message_id) VALUES (?1, ?2, ?3)
                 ON CONFLICT(case_id) DO UPDATE SET chat_id=excluded.chat_id, message_id=excluded.message_id",
                params![case_id, chat_id, message_id],
            )?;
            Ok(())
        })
        .await
    }

    /// Returns and removes the stored confirmation location for a case - it
    /// is only edited once, so the row is consumed on read.
    async fn take_report_confirmation(&self, case_id: &str) -> Option<(i64, i32)> {
        let case_id = case_id.to_string();
        self.with_conn(move |conn| {
            let found = conn
                .query_row(
                    "SELECT chat_id, message_id FROM report_confirmations WHERE case_id = ?1",
                    params![&case_id],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i32>(1)?)),
                )
                .ok();
            if found.is_some() {
                conn.execute("DELETE FROM report_confirmations WHERE case_id = ?1", params![&case_id])?;
            }
            Ok(found)
        })
        .await
        .ok()
        .flatten()
    }

    /// `/spam` three-strike rule. A third of reports were being rejected,
    /// and Telegram won't let the bot DM a reporter who has never messaged
    /// it, so there's no way to tell someone their report was wrong. Losing
    /// the command after three rejections is the feedback instead.
    ///
    /// Counted here rather than derived from `cases` because rejecting a
    /// report overwrites that case's `actor_user_id` with the *reviewer*,
    /// so the reporter is no longer recorded on it afterwards.
    async fn report_strikes(&self, user_id: i64) -> i64 {
        self.with_conn(move |conn| {
            Ok(conn
                .query_row("SELECT rejected_count FROM report_offenses WHERE user_id = ?1", params![user_id], |row| row.get(0))
                .unwrap_or(0))
        })
        .await
        .unwrap_or(0)
    }

    async fn add_report_strike(&self, user_id: i64) -> Result<i64> {
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO report_offenses (user_id, rejected_count, last_rejected_at) VALUES (?1, 1, ?2)
                 ON CONFLICT(user_id) DO UPDATE SET rejected_count = rejected_count + 1, last_rejected_at = excluded.last_rejected_at",
                params![user_id, Utc::now().to_rfc3339()],
            )?;
            Ok(conn.query_row("SELECT rejected_count FROM report_offenses WHERE user_id = ?1", params![user_id], |row| row.get(0))?)
        })
        .await
    }

    async fn clear_report_strikes(&self, user_id: i64) -> Result<()> {
        self.with_conn(move |conn| {
            conn.execute("DELETE FROM report_offenses WHERE user_id = ?1", params![user_id])?;
            Ok(())
        })
        .await
    }

    /// The permission predicate everything gates on: the host, or a
    /// command-granted maintainer. Replaces the old env-var check and the
    /// old `is_special_user` (they were the same set, so they are collapsed
    /// into one). Also the moderation-exemption test - a maintainer is never
    /// auto-banned.
    async fn is_maintainer(&self, user_id: i64) -> bool {
        is_host(user_id) || self.maintainers.read().await.contains(&user_id)
    }

    async fn set_maintainer(&self, user_id: i64, enabled: bool, added_by: Option<i64>) -> Result<()> {
        self.with_conn(move |conn| {
            if enabled {
                conn.execute(
                    "INSERT OR IGNORE INTO maintainers (user_id, added_by, created_at) VALUES (?1, ?2, ?3)",
                    params![user_id, added_by, Utc::now().to_rfc3339()],
                )?;
            } else {
                conn.execute("DELETE FROM maintainers WHERE user_id = ?1", params![user_id])?;
            }
            Ok(())
        })
        .await?;
        let mut cache = self.maintainers.write().await;
        if enabled {
            cache.insert(user_id);
        } else {
            cache.remove(&user_id);
        }
        Ok(())
    }

    /// Granted maintainers only (the host is a constant, not a row here).
    async fn list_maintainers(&self) -> Result<Vec<(i64, String)>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare("SELECT user_id, created_at FROM maintainers ORDER BY created_at ASC")?;
            let mut rows = stmt.query([])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push((row.get(0)?, row.get(1)?));
            }
            Ok(out)
        })
        .await
    }

    /// Reviewer-role checks. Read straight from SQLite rather than cached
    /// like the denial lists: these are only consulted when someone presses
    /// a button in the report channel, not on every message, so there's no
    /// hot path to protect and no cache to keep in sync.
    async fn is_reviewer(&self, user_id: i64) -> bool {
        self.with_conn(move |conn| {
            let count: i64 = conn.query_row("SELECT COUNT(*) FROM reviewers WHERE user_id = ?1", params![user_id], |row| row.get(0))?;
            Ok(count > 0)
        })
        .await
        .unwrap_or(false)
    }

    async fn set_reviewer(&self, user_id: i64, enabled: bool, added_by: Option<i64>) -> Result<()> {
        self.with_conn(move |conn| {
            if enabled {
                conn.execute(
                    "INSERT OR IGNORE INTO reviewers (user_id, added_by, created_at) VALUES (?1, ?2, ?3)",
                    params![user_id, added_by, Utc::now().to_rfc3339()],
                )?;
            } else {
                conn.execute("DELETE FROM reviewers WHERE user_id = ?1", params![user_id])?;
            }
            Ok(())
        })
        .await
    }

    async fn list_reviewers(&self) -> Result<Vec<(i64, String)>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare("SELECT user_id, created_at FROM reviewers ORDER BY created_at ASC")?;
            let mut rows = stmt.query([])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push((row.get(0)?, row.get(1)?));
            }
            Ok(out)
        })
        .await
    }

    /// Both denial lists for `/list_banned`, as (id, reason, created_at).
    /// Reads from SQLite rather than the in-memory sets, since only the DB
    /// keeps the reason and timestamp - the caches hold ids alone.
    #[allow(clippy::type_complexity)]
    async fn list_banned(&self) -> Result<(Vec<(i64, String, String)>, Vec<(i64, String, String)>)> {
        self.with_conn(|conn| {
            fn rows(conn: &Connection, sql: &str) -> rusqlite::Result<Vec<(i64, String, String)>> {
                let mut stmt = conn.prepare(sql)?;
                let mut out = Vec::new();
                let mut rows = stmt.query([])?;
                while let Some(row) = rows.next()? {
                    out.push((row.get(0)?, row.get(1)?, row.get(2)?));
                }
                Ok(out)
            }
            let groups = rows(conn, "SELECT chat_id, reason, created_at FROM banned_groups ORDER BY created_at DESC")?;
            let users = rows(conn, "SELECT user_id, reason, created_at FROM banned_users ORDER BY created_at DESC")?;
            Ok((groups, users))
        })
        .await
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

    /// Generic maintainer-controlled gate: has `module` been granted to
    /// `chat_id` via `/magic`? Only gates *enabling* a module through
    /// `/module` - turning one off is never gated. Not specific to `pol`,
    /// so a future customized module reuses this same table.
    async fn is_module_allowed(&self, module: &str, chat_id: i64) -> Result<bool> {
        let module = module.to_string();
        self.with_conn(move |conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM module_allowlist WHERE module = ?1 AND chat_id = ?2",
                params![module, chat_id],
                |row| row.get(0),
            )?;
            Ok(count > 0)
        })
        .await
    }

    async fn set_module_allowed(&self, module: &str, chat_id: i64, enabled: bool, added_by: Option<i64>) -> Result<()> {
        let module = module.to_string();
        self.with_conn(move |conn| {
            if enabled {
                conn.execute(
                    "INSERT OR IGNORE INTO module_allowlist (module, chat_id, added_by, created_at) VALUES (?1, ?2, ?3, ?4)",
                    params![module, chat_id, added_by, Utc::now().to_rfc3339()],
                )?;
            } else {
                conn.execute(
                    "DELETE FROM module_allowlist WHERE module = ?1 AND chat_id = ?2",
                    params![module, chat_id],
                )?;
            }
            Ok(())
        })
        .await
    }

    /// Current warn count for (chat_id, user_id) under the "warn-pol"
    /// module - 0 if they've never been warned there. Never expires,
    /// unlike `permission_offenses`.
    async fn pol_warn_count(&self, chat_id: i64, user_id: i64) -> Result<i64> {
        self.with_conn(move |conn| {
            let count: Option<i64> = conn
                .query_row(
                    "SELECT warn_count FROM pol_warnings WHERE chat_id = ?1 AND user_id = ?2",
                    params![chat_id, user_id],
                    |row| row.get(0),
                )
                .ok();
            Ok(count.unwrap_or(0))
        })
        .await
    }

    /// Records a warning (or ban) event and returns the new total count.
    async fn increment_pol_warn(&self, chat_id: i64, user_id: i64) -> Result<i64> {
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO pol_warnings (chat_id, user_id, warn_count, last_warned_at) VALUES (?1, ?2, 1, ?3)
                 ON CONFLICT(chat_id, user_id) DO UPDATE SET warn_count = warn_count + 1, last_warned_at = excluded.last_warned_at",
                params![chat_id, user_id, Utc::now().to_rfc3339()],
            )?;
            let count: i64 = conn.query_row(
                "SELECT warn_count FROM pol_warnings WHERE chat_id = ?1 AND user_id = ?2",
                params![chat_id, user_id],
                |row| row.get(0),
            )?;
            Ok(count)
        })
        .await
    }

    async fn clear_pol_warns(&self, chat_id: i64, user_id: i64) -> Result<()> {
        self.with_conn(move |conn| {
            conn.execute("DELETE FROM pol_warnings WHERE chat_id = ?1 AND user_id = ?2", params![chat_id, user_id])?;
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
        let mut no_halal = Vec::new();

        if settings.no_halal && !self.is_maintainer(user.id.0 as i64).await {
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

        Ok(ModuleCheckResult { reasons, name_guard: Vec::new(), no_halal })
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
    HostCtl(String),
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
    MlExport,
    MlPurge(String),
    MlPurgeText(String),
    MlRebuild,
    MlRetrain,
    MlDedupe,
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
    RefreshBL,
    Forbid(String),
    Forgive(String),
    ListBanned,
    Reviewer(String, String),
    Maintainer(String, String),
    Whois(String),
    ReportReset(String),
    MlEval(String),
    ListRules,
    CheckRules,
    DelRule(String),
    Module(String, String),
    White(String),
    Unwhite(String),
    WhiteGlobal(String),
    UnwhiteGlobal(String),
    HelpOp(String),
    Check(String),
    Unban(String),
    Unmute(String),
    Ping,
    SetAuditLog(String),
    Revert(String),
    SetExchangeChannel(String),
    Magic(String, String, String),
    Pol(String),
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
        "/hostctl" => ModerationCommand::HostCtl(text.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim().to_string()),
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
        "/ml_export" => ModerationCommand::MlExport,
        "/ml_purge" => ModerationCommand::MlPurge(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/ml_purge_text" => ModerationCommand::MlPurgeText(text.split_whitespace().skip(1).collect::<Vec<_>>().join(" ")),
        "/ml_rebuild" => ModerationCommand::MlRebuild,
        "/ml_retrain" => ModerationCommand::MlRetrain,
        "/ml_dedupe" => ModerationCommand::MlDedupe,
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
        "/refreshbl" => ModerationCommand::RefreshBL,
        "/forbid" => ModerationCommand::Forbid(text.split_whitespace().skip(1).collect::<Vec<_>>().join(" ")),
        "/forgive" => ModerationCommand::Forgive(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/list_banned" => ModerationCommand::ListBanned,
        "/whois" => ModerationCommand::Whois(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/report_reset" => ModerationCommand::ReportReset(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/ml_eval" => ModerationCommand::MlEval(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/reviewer" => ModerationCommand::Reviewer(
            text.split_whitespace().nth(1).unwrap_or("").to_string(),
            text.split_whitespace().nth(2).unwrap_or("").to_string(),
        ),
        "/maintainer" => ModerationCommand::Maintainer(
            text.split_whitespace().nth(1).unwrap_or("").to_string(),
            text.split_whitespace().nth(2).unwrap_or("").to_string(),
        ),
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
        "/help_op" => ModerationCommand::HelpOp(text.split_whitespace().nth(1).unwrap_or("").to_string()),
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
        "/set_exchange_channel" => ModerationCommand::SetExchangeChannel(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        "/magic" => {
            let mut parts = text.split_whitespace();
            let _ = parts.next();
            let module = parts.next().unwrap_or("").to_string();
            let chat_id = parts.next().unwrap_or("").to_string();
            let action = parts.next().unwrap_or("").to_string();
            ModerationCommand::Magic(module, chat_id, action)
        }
        "/pol" => ModerationCommand::Pol(text.split_whitespace().nth(1).unwrap_or("").to_string()),
        _ => ModerationCommand::Unknown,
    }
}

/// The provenance markers `extract_full_text` appends, as a single token
/// each, instead of as prose.
///
/// Those markers are written for humans reading a `/case` - `[fwd_id: -100…]`,
/// `[external_origin_username: SOMECHANNEL]` - but they were being fed to the
/// tokenizer verbatim, and `_` counts as punctuation, so a key like
/// `external_reply_origin_channel_id` shattered into `external`, `reply`,
/// `origin`, `channel`, `id`. Training one forwarded advert therefore taught
/// the model that those words are spam. Two ways that misfires:
///
/// - They are attached to *every* forward and external reply regardless of
///   content, so all forwarded messages drift toward spam together.
/// - `reply`, `chat`, `id`, `username`, `channel` are ordinary English. A
///   member writing "please reply in the chat, what's your username" tokenizes
///   to three of the exact tokens the advert trained.
///
/// The identity itself is real signal - a forward from a known spam channel
/// says something - so it is kept, as one opaque `tgsrc…` token that cannot
/// collide with prose. The key names are dropped entirely.
fn collapse_origin_markers(text: &str) -> String {
    static MARKER_RE: OnceLock<StdRegex> = OnceLock::new();
    let re = MARKER_RE.get_or_init(|| {
        StdRegex::new(
            r"\[(?:fwd_id|fwd_user|external_origin_chat_id|external_origin_username|external_reply_origin_channel_id|external_reply_origin_channel_username|external_reply_origin_chat_id|external_reply_origin_chat_username):\s*([^\]]*)\]",
        )
        .expect("valid origin marker regex")
    });
    re.replace_all(text, |caps: &regex::Captures| {
        let value: String = caps[1].chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        if value.is_empty() {
            " ".to_string()
        } else {
            format!(" tgsrc{} ", value.to_lowercase())
        }
    })
    .into_owned()
}

fn tokenize(text: &str) -> Vec<String> {
    normalize_tokens(&collapse_origin_markers(text), jieba())
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
    // Layout notes, learned the hard way: `<code>` renders monospace, and
    // monospace CJK is wide, so a line that looks short in source wraps on a
    // phone. Ideographic spaces (U+3000) as pseudo-columns make it worse -
    // nothing aligns, and a wrapped line loses the indent entirely. So:
    // one item per line, a single ASCII space before the description, and
    // short enough that nothing wraps at default phone width. Modules are
    // grouped by default state rather than repeating "預設開啟" on each row.
    concat!(
        "<b>Spam Protection Bot（SPB）</b>\n",
        "全自動反廣告機器人。加進群組後請給予管理員權限（至少「刪除訊息」與「封禁用戶」），即會自動運作。\n",
        "\n<b>━━ 所有人 ━━</b>\n",
        "<code>/spam</code> 回覆可疑訊息提交舉報\n",
        "<code>/case &lt;ID&gt;</code> 查詢封禁記錄\n",
        "<code>/id</code> 取得自己的 User ID\n",
        "· 舉報累計 3 次被拒將暫停使用\n",
        "\n<b>━━ 群組管理員 ━━</b>\n",
        "<code>/sb</code> 刪除訊息並封禁\n",
        "<code>/mute</code> 禁言\n",
        "<code>/kick</code> 踢出\n",
        "<code>/unban</code> <code>/unmute</code> 解除（僅本群）\n",
        "<code>/white</code> <code>/unwhite</code> 本群白名單\n",
        "· 加 <code>-global</code> 為全域白名單\n",
        "<code>/module</code> 查看模組開關狀態\n",
        "<code>/module 名稱 on|off</code> 切換\n",
        "<code>/module all on|off</code> 全開／全關\n",
        "\n<b>━━ 模組 ━━</b>\n",
        "<b>預設開啟</b>\n",
        "· Flood 洗版偵測\n",
        "· GuestBan 訪客機器人廣告\n",
        "<b>需自行開啟</b>\n",
        "· NoHalal 清真內容檢查\n",
        "· NoSM 自動刪除服務訊息\n",
        "· Captcha 入群驗證\n",
        "· Netban 訂閱專案黑名單\n",
        "· CmdClean 指令濫用防護\n",
        "· NoContact 禁止分享聯絡人\n",
        "· NoVoice 禁止語音訊息\n",
        "· NoExec 禁止可執行檔\n",
        "\n基礎防護（Flood、GuestBan）不會被 <code>/module all off</code> 關閉。\n",
        "\n<b>━━ 被封禁後 ━━</b>\n",
        "用 <code>/id</code> 取得 User ID，到 @SpamProtectionLogging 搜尋，或向 @SEELE_01_BOT 申訴。\n",
        "\n交流群 @SpamProtectionChat\n",
        "日誌 @SpamProtectionLogging",
    )
    .to_string()
}

/// Maintainer help, split into sections.
///
/// It used to be one string listing every command with a full paragraph of
/// explanation each. That had already reached 3,000 of Telegram's 4,096
/// character limit, so it would eventually have started failing to send
/// outright, and it was unreadable long before that.
///
/// Same layout rules as `help_text`: one command per line, a single ASCII
/// space before the description, no ideographic-space columns, and lines
/// short enough not to wrap on a phone once `<code>` widens them.
fn help_op_text(section: &str) -> String {
    match section.trim().to_lowercase().as_str() {
        "ml" | "model" | "模型" => concat!(
            "<b>━━ 模型 / 訓練 ━━</b>\n",
            "\n<b>查看</b>\n",
            "<code>/ml_stats</code> 樣本量與有效門檻\n",
            "<code>/ml_score</code> 測試單條文本分數\n",
            "<code>/ml_score_debug</code> 逐詞看分數怎麼算\n",
            "<code>/ml_eval [比例]</code> 留出法評估\n",
            "· 列出各門檻的精確率／召回率／F1\n",
            "· 只讀，不會改動實際模型\n",
            "<code>/ml_threshold &lt;值&gt;</code> 調整門檻\n",
            "· 私訊／測試群／工作群＝全域\n",
            "· 其他群組＝僅該群\n",
            "\n<b>訓練</b>\n",
            "<code>/ml_train_spam</code> 標記為 spam\n",
            "<code>/mark_ham</code> 標記為正常\n",
            "<code>/ml_clean_spam</code> 清成 ham\n",
            "<code>/ml_undo_clean_spam</code> 撤銷上一步\n",
            "<code>/import</code> 匯入訓練列表\n",
            "<code>/ml_export</code> 匯出訓練資料\n",
            "\n<b>批量（限私訊）</b>\n",
            "<code>/ml_start_mass_train_smart</code>\n",
            "<code>/ml_start_mass_train_plain</code>\n",
            "<code>/ml_finish_mass_train</code>\n",
            "<code>/ml_start_mass_ham</code>\n",
            "<code>/ml_finish_mass_ham</code>\n",
            "\n<b>維護</b>\n",
            "<code>/ml_purge &lt;case_id&gt;</code> 刪誤樣本\n",
            "<code>/ml_purge_text &lt;片段&gt;</code> 依文字刪\n",
            "<code>/ml_dedupe</code> 合併重複、移除空白\n",
            "<code>/ml_rebuild</code> 從資料庫重載\n",
            "<code>/ml_retrain</code> 重算全部詞頻\n",
            "· 分詞規則改變後要跑這個\n",
            "· <code>/ml_rebuild</code> 不會修正舊詞頻\n",
        ).to_string(),

        "user" | "access" | "權限" => concat!(
            "<b>━━ 用戶與權限 ━━</b>\n",
            "\n<b>查詢與解除</b>\n",
            "<code>/whois &lt;user_id&gt;</code> 完整紀錄\n",
            "· 身分、歷史封禁次數\n",
            "· 目前所有生效中的封禁\n",
            "· 跨群組黑名單、舉報被拒次數\n",
            "<code>/unban</code> 維護組完整版\n",
            "· 回覆／user_id／case_id 皆可\n",
            "· 一併移除誤訓練樣本並重建模型\n",
            "· 曾 Netban 同步的群組也會解封\n",
            "<code>/unmute</code> 解除禁言並撤銷案例\n",
            "\n<b>項目層級封禁</b>\n",
            "<code>/forbid &lt;user_id&gt; [原因]</code>\n",
            "· 該帳號所有指令不予回應\n",
            "· 且無法把機器人加進任何群組\n",
            "<code>/forgive &lt;id&gt;</code> 解除\n",
            "· 負數＝群組，正數＝用戶\n",
            "<code>/list_banned</code> 列出封禁名單\n",
            "\n<b>審核員</b>\n",
            "<code>/reviewer add|del &lt;user_id&gt;</code>\n",
            "<code>/reviewer list</code>\n",
            "· 只能處理舉報頻道的按鈕\n",
            "· 維護組不需另外授予\n",
            "<code>/report_reset &lt;user_id&gt;</code>\n",
            "· 清除舉報被拒計數（滿 3 次暫停）\n",
        ).to_string(),

        "group" | "ops" | "群組" => concat!(
            "<b>━━ 群組與運維 ━━</b>\n",
            "\n<b>群組</b>\n",
            "<code>/setchat [chat_id]</code> 設定工作群\n",
            "· 會自動取消頻道發文的自動釘選\n",
            "<code>/leave [chat_id] [原因]</code> 終止服務\n",
            "· 發出終止通知並離開該群\n",
            "· 列入封禁名單，加回會自動退出\n",
            "<code>/pol show</code> 查詢本群警告次數\n",
            "<code>/pol clear</code> 清除警告\n",
            "<code>/ping</code> 版本與 commit\n",
            "\n<b>日誌與橋接</b>\n",
            "<code>/set_audit_log [chat_id]</code>\n",
            "· 記錄每個改變狀態的維護指令\n",
            "<code>/revert &lt;action_id&gt;</code> 復原操作\n",
            "· 少數操作無法自動復原\n",
            "<code>/set_exchange_channel &lt;chat_id&gt;</code>\n",
            "· PM 申訴橋接頻道\n",
            "\n<b>封禁代號說明</b>\n",
            "<code>/updateBL</code> 重新發文並釘選\n",
            "<code>/refreshBL</code> 就地編輯上一則\n",
        ).to_string(),

        "rules" | "規則" => concat!(
            "<b>━━ 正則規則 ━━</b>\n",
            "<code>/add_rule &lt;regex&gt;</code> 新增\n",
            "· 會再追問規則名稱\n",
            "<code>/edit_rule &lt;id&gt; &lt;regex&gt;</code> 改正則\n",
            "<code>/del_rule &lt;id&gt;</code> 刪除\n",
            "<code>/list_rules</code> 列出目前規則\n",
            "<code>/check_rules</code> 列出無法編譯的\n",
            "\n· 規則命中即視為 spam（分數 1.0）\n",
            "· 會進入跨群組黑名單\n",
        ).to_string(),

        _ => concat!(
            "<b>❖ 維護指令</b>\n",
            "<code>/help_op &lt;分類&gt;</code> 查看完整說明\n",
            "\n<b>━━ 分類 ━━</b>\n",
            "<code>/help_op ml</code> 模型與訓練\n",
            "<code>/help_op user</code> 用戶與權限\n",
            "<code>/help_op group</code> 群組與運維\n",
            "<code>/help_op rules</code> 正則規則\n",
            "\n<b>━━ 最常用 ━━</b>\n",
            "<code>/whois &lt;user_id&gt;</code> 某人的所有紀錄\n",
            "<code>/unban</code> 解封並清除誤樣本\n",
            "<code>/ml_eval</code> 評估模型、決定門檻\n",
            "<code>/ml_stats</code> 樣本量與門檻\n",
            "<code>/case &lt;ID&gt;</code> 查詢單一案例\n",
            "<code>/revert &lt;action_id&gt;</code> 復原操作\n",
            "\n<b>━━ 備註 ━━</b>\n",
            "· 這裡只列維護者指令\n",
            "· 群組管理員指令見 <code>/help</code>\n",
            "· 審核員僅能處理舉報頻道按鈕\n",
        ).to_string(),
    }
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

/// The publicly-documented modules, in the order `/module` lists them, as
/// (lookup key, display name, is_baseline). Deliberately excludes
/// "warn-pol" - that one is maintainer-gated and must never surface in any
/// user-facing list or bulk toggle (see GroupModuleSettings::pol).
///
/// `is_baseline` marks the two that default on (see GroupModuleSettings'
/// own defaults): they're protection every group is meant to have, not an
/// opinionated choice, so `/module all off` leaves them alone rather than
/// quietly stripping a group's spam defences in one keystroke. Turning
/// either off individually still works and is unaffected.
const PUBLIC_MODULES: &[(&str, &str, bool)] = &[
    ("nohalal", "NoHalal", false),
    ("nosm", "NoSM", false),
    ("flood", "Flood", true),
    ("captcha", "Captcha", false),
    ("netban", "Netban", false),
    ("cmdclean", "CmdClean", false),
    ("guestban", "GuestBan", true),
    ("nocontact", "NoContact", false),
    ("novoice", "NoVoice", false),
    ("noexec", "NoExec", false),
];

/// Current value of one module flag by its `/module` key, or `None` if the
/// key isn't a real module - which is also how the command arm tells a typo
/// from a valid name.
fn module_flag(settings: &GroupModuleSettings, key: &str) -> Option<bool> {
    Some(match key {
        "nohalal" => settings.no_halal,
        "nosm" => settings.no_service_messages,
        "flood" => settings.flood_control,
        "captcha" => settings.captcha,
        "netban" => settings.netban,
        "cmdclean" => settings.cmd_clean,
        "guestban" => settings.guest_ban,
        "nocontact" => settings.no_contact,
        "novoice" => settings.no_voice,
        "noexec" => settings.no_exec,
        "warn-pol" => settings.pol,
        _ => return None,
    })
}

/// Removes every `@username` token, leaving whatever else was typed. Backs
/// `find_recent_guest_invoker`'s "is this message nothing but mentions?"
/// test. Telegram usernames are ASCII alphanumeric plus underscore, so the
/// run after an `@` ends at the first character outside that set - which
/// leaves CJK text, emoji, and punctuation in the remainder, exactly the
/// content that should disqualify a message from looking like a bare summon.
/// If `text` is nothing but one or more bot @-mentions (ignoring emoji,
/// whitespace and punctuation), returns the bot usernames, lowercased and
/// without the @. This is the shape of guest-mode summon spam: a bare
/// "@somebot", or "@a_bot @b_bot 🎉". Returns None the moment there is any
/// real word content, or a mention that isn't a bot (Telegram bot
/// usernames end in "bot").
fn bot_mentions_only(text: &str) -> Option<Vec<String>> {
    let mut bots = Vec::new();
    let mut leftover = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '@' {
            let mut name = String::new();
            while chars.peek().is_some_and(|n| n.is_ascii_alphanumeric() || *n == '_') {
                name.push(chars.next().unwrap());
            }
            if name.is_empty() {
                leftover.push('@');
            } else {
                bots.push(name.to_lowercase());
            }
        } else {
            leftover.push(c);
        }
    }
    if bots.is_empty() {
        return None;
    }
    if !bots.iter().all(|b| b.ends_with("bot")) {
        return None;
    }
    // Any letter or digit left over (CJK included) means real text, not a
    // bare summon. Emoji and punctuation are fine.
    if leftover.chars().any(|c| c.is_alphanumeric()) {
        return None;
    }
    let mut seen = std::collections::HashSet::new();
    Some(bots.into_iter().filter(|b| seen.insert(b.clone())).collect())
}

/// Turns a pure bot-mention message into regex rules - one per handle - so
/// the bot is caught deterministically wherever it appears next. Shared by
/// the manual /sb path and the auto-ban path (a member pasting bot handles
/// as an advert scores as spam and lands here too, not just guest-mode
/// summons). Logs each created rule to the log channel for review. Returns
/// true if `evidence` was a pure bot mention at all (whether or not new
/// rules were created), so the caller can label the case BOTSPAM and skip
/// ML training on a bare username.
async fn capture_bot_spam_rules(bot: &Bot, runtime: &Runtime, chat_id: i64, evidence: &str) -> bool {
    let Some(bots) = bot_mentions_only(evidence) else { return false; };
    let mut created = Vec::new();
    for b in &bots {
        let pattern = format!("(?i)@{b}\\b");
        if runtime.spam_rule_pattern_exists(&pattern).await {
            continue;
        }
        if let Ok(id) = runtime
            .add_spam_rule(&pattern, &format!("純機器人提及 spam：@{b}（自動建立）"))
            .await
        {
            created.push((id, b.clone()));
        }
    }
    if !created.is_empty() {
        let list = created
            .iter()
            .map(|(id, b)| format!("@{b}（規則 #{id}）"))
            .collect::<Vec<_>>()
            .join("、");
        let _ = bot
            .send_message(
                ChatId(runtime.config.log_channel_id),
                format!("<b>已自動建立機器人提及規則</b>\n來源：群組 <code>{chat_id}</code>\n{list}\n如為誤判請用 /del_rule 移除。"),
            )
            .parse_mode(ParseMode::Html)
            .await;
    }
    true
}

fn strip_mentions(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '@' {
            out.push(c);
            continue;
        }
        while chars.peek().is_some_and(|n| n.is_ascii_alphanumeric() || *n == '_') {
            chars.next();
        }
    }
    out
}

fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// A clickable mention that works even without a reply to hook onto (the
/// original message is already gone by the time a `/pol` warning sends) and
/// even for users with no @username, since it targets a user_id directly.
fn mention_link(user_id: i64, name: &str) -> String {
    format!("<a href=\"tg://user?id={user_id}\">{}</a>", escape_html(name))
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
            ActionKind::GuestBotBan => "訪客模式機器人封禁".to_string(),
            ActionKind::GuestInvokerBan => "訪客模式召喚者封禁".to_string(),
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
    "<b>❖ 封禁代號說明</b>\n\n- <code>ARABIC</code>: 偵測到清真\n- <code>REGEX</code>: 觸發正則規則\n- <code>FLOOD</code>: 洗版偵測（5 秒內傳送 5 條以上訊息）\n- <code>PERM_REPEAT</code>: 24 小時內重複嘗試使用無權限的指令\n- <code>GUEST_MODE</code>: 訪客模式機器人（未加入本群卻發文）\n- <code>GUEST_MODE_INVOKER</code>: 召喚訪客模式機器人的人\n- <code>CONTACT</code>: 在群組分享聯絡人\n- <code>VOICE</code>: 傳送語音訊息\n- <code>EXEC_FILE</code>: 傳送可執行檔案\n- <code>ML</code>: 機器學習模型判定為垃圾訊息\n- <code>BOTSPAM</code>: 純機器人提及（訪客模式召喚廣告）\n\n申訴找 @SEELE_01_BOT".to_string()
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


/// Guards a command handler arm behind `is_maintainer`, replying with `$msg` and
/// returning early otherwise. Collapses the same 4-line permission check that
/// used to be repeated at the top of ~24 command arms in `handle_command`.
macro_rules! require_maintainer {
    ($bot:expr, $runtime:expr, $from_id:expr, $message:expr, $msg:expr) => {
        if !$runtime.is_maintainer($from_id).await {
            $bot.send_message($message.chat.id, $msg).await?;
            return Ok(());
        }
    };
}

/// Whether `user_id` is an owner or administrator of `chat_id`.
///
/// Uses teloxide's own `is_privileged()` rather than matching text. This
/// used to be `format!("{:?}", member).contains("Administrator")`, which
/// formatted the *whole* `ChatMember` - including the user's profile - and
/// searched that. Anyone could therefore become an administrator here by
/// setting their first name to "Owner": the debug string contained the
/// word, so the check passed while `kind` was plainly `Member`. That
/// granted the full group-admin command set (`/sb`, `/mute`, `/kick`,
/// `/module`, `/white`, ...) to any member who renamed themselves.
async fn is_group_admin(bot: &Bot, chat_id: ChatId, user_id: i64) -> bool {
    match bot.get_chat_member(chat_id, UserId(user_id as u64)).await {
        Ok(member) => member.kind.is_privileged(),
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


/// The project host (項目主持人). Hardcoded, singular, and unremovable by any
/// command - the root of the permission hierarchy. Everyone else's
/// authority is granted downward from here: the host grants maintainers,
/// maintainers grant reviewers. This being a source constant rather than a
/// database row or an env var is deliberate - it is the one authority that
/// must survive a wiped database, a changed environment, or a fork of this
/// code (see the note answering that question in the project history).
const HOST_ID: i64 = 5172206551;

fn is_host(user_id: i64) -> bool {
    user_id == HOST_ID
}

/// Telegram's own reserved pseudo-accounts, not real chat members and never
/// bannable in any way that matters. 777000 ("Telegram") is reused as the
/// immediate sender for messages auto-forwarded from a linked channel into
/// its discussion group (and for platform service notifications) - a group
/// whose channel is linked will see routine, legitimate channel content
/// arrive "from" this id constantly. 1087968824 ("GroupAnonymousBot") and
/// 136817688 ("Channel") are the pseudo-senders for anonymous-admin posts
/// and anonymous channel posts respectively. None of these identify an
/// actual person to hold accountable, so every message-time moderation
/// check must treat them like `is_special_user` - an incident where the
/// classifier auto-banned 777000 for a forwarded post's content, and the
/// same-chat reban safety net then kept re-deleting every legitimate
/// announcement that followed, is exactly why this exists.
fn is_platform_pseudo_user(user_id: i64) -> bool {
    const IDS: &[i64] = &[777000, 1087968824, 136817688];
    IDS.contains(&user_id)
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
        if let Ok(sent) = bot
            .send_message(message.chat.id, format!("✅ {} 驗證通過，歡迎！", escape_html(&short_user(user))))
            .parse_mode(ParseMode::Html)
            .await
        {
            // Clear the welcome after 30s, same self-delete pattern as
            // notify_group - it's a transient confirmation, not chat history.
            let bot = bot.clone();
            let chat_id = message.chat.id;
            let sent_id = sent.id;
            tokio::spawn(async move {
                sleep(Duration::from_secs(30)).await;
                let _ = bot.delete_message(chat_id, sent_id).await;
            });
        }
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

    // Only notify when this bot itself is the one joining - not any other
    // bot a group happens to add alongside it. `is_bot` alone matches any
    // bot account, which was wrongly firing this for third-party bots too.
    if let Some(me_id) = runtime.me_id(bot).await {
        if users.iter().any(|u| u.id == me_id) {
            // "may not participate in, use, operate or otherwise access the
            // project's services" - enforced at the one moment it can be:
            // whoever pulled the bot in is named on this very message, so a
            // forbidden account can't route around their ban by inviting it
            // somewhere new. The group's own ban is handled by the dispatcher
            // guard, but is repeated here so the notice gets posted before
            // leaving rather than the bot silently vanishing.
            let inviter_banned = match message.from.as_ref() {
                Some(user) => runtime.is_user_banned(user.id.0 as i64).await,
                None => false,
            };
            // Join/leave notices go to the audit log (host-only) rather than
            // the report channel; fall back to the report channel only if no
            // audit log is configured.
            let notice_dest = runtime.audit_log_chat().await.unwrap_or(runtime.config.report_channel_id);
            if inviter_banned || runtime.is_group_banned(message.chat.id.0).await {
                let _ = bot.send_message(message.chat.id, service_denied_text()).parse_mode(ParseMode::Html).await;
                let _ = bot.leave_chat(message.chat.id).await;
                let text = format!(
                    "<b>已拒絕加入</b>\n<b>群組</b>: <code>{}</code>\n<b>標題</b>: {}\n<b>來源</b>: <code>{}</code>\n<b>原因</b>: {}",
                    message.chat.id.0,
                    escape_html(message.chat.title().unwrap_or("unknown")),
                    message.from.as_ref().map(short_user).unwrap_or_else(|| "unknown".to_string()),
                    if inviter_banned { "邀請者已被禁止使用本項目" } else { "此群組已被終止服務" },
                );
                let _ = bot.send_message(ChatId(notice_dest), text).parse_mode(ParseMode::Html).await;
                return true;
            }

            let title = message.chat.title().unwrap_or("unknown");
            let text = format!(
                "<b>機器人已加入</b>\n<b>群組</b>: <code>{}</code>\n<b>標題</b>: {}\n<b>來源</b>: <code>{}</code>",
                message.chat.id.0,
                escape_html(title),
                message.from.as_ref().map(short_user).unwrap_or_else(|| "unknown".to_string())
            );
            let _ = bot.send_message(ChatId(notice_dest), text).parse_mode(ParseMode::Html).await;

            // Not self-deleting, unlike the transient group notices: this is
            // the group's one prompt to read what it just agreed to, and it
            // should still be there for anyone scrolling back.
            let _ = bot
                .send_message(message.chat.id, welcome_text())
                .parse_mode(ParseMode::Html)
                .reply_markup(terms_button("閱讀使用規範 / Terms of Use"))
                .await;
        }
    }

    for user in users {
        // Never name-guard a bot account - including this one. The name
        // guard is written for human display names, and "Spam Protection"
        // itself trips NL13 (two segments, >=13 chars) and NLTAIL, so
        // joining a group with NoLongName on made the bot ban *itself* -
        // which Telegram enforces by removing it from the group, so it
        // looked like the bot spontaneously left. Every other moderation
        // path here already skips `is_bot`; this loop was the exception.
        if user.is_bot {
            continue;
        }
        if runtime.is_maintainer(user.id.0 as i64).await || is_platform_pseudo_user(user.id.0 as i64) {
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

        if enabled.no_halal {
            let all_reasons = {
                let profile = runtime.load_user_profile(bot, user.id.0 as i64).await.ok();
                let bio = profile.as_ref().and_then(|p| p.bio.as_deref());
                evaluate_module_checks(user, user.username.as_deref(), bio, None)
            };

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
                broadcast_ban_status(bot, runtime, updated.target_user_id, true).await;
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
                notify_netban_sync(bot, message.chat.id, user.id.0 as i64, &prior_case.id).await;
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

/// Public Terms of Use. Points at the GitHub Pages site rather than the
/// repository: users being told the rules have no reason to be handed the
/// source tree, and the published page is the authoritative wording.
const TERMS_URL: &str = "https://wikipedia-zh.github.io/Anti-Spam/";

fn terms_button(label: &str) -> InlineKeyboardMarkup {
    InlineKeyboardMarkup::new(vec![vec![InlineKeyboardButton::url(
        label.to_string(),
        Url::parse(TERMS_URL).expect("TERMS_URL is a valid literal URL"),
    )]])
}

/// Posted to a group the moment the bot is added. The terms bind the whole
/// group once it starts using the service, so this is the one point where
/// everyone present can reasonably be expected to see them - a link buried
/// in `/help` would not be.
fn welcome_text() -> String {
    "<b>Spam Protection Bot（SPB）已加入本群</b>\n\n本機器人會自動偵測並處理垃圾訊息。請群組管理員給予「刪除訊息」與「封禁使用者」權限，否則無法正常運作。\n\n使用本機器人即表示同意本項目的使用規範，請點擊下方按鈕閱讀。管理員可用 <code>/module</code> 查看與調整各項功能，<code>/help</code> 查看指令說明。".to_string()
}

/// A ground for `/leave`, resolved to a display label and the Terms-of-Use
/// section it points at. A maintainer types a short code after the chat id
/// (`/leave -100… TnS`); a known code links straight to the relevant clause,
/// anything else is carried through as free-text with no specific anchor.
struct TerminationReason {
    label: String,
    anchor: Option<&'static str>,
}

/// Turns the raw reason typed after `/leave` into a `TerminationReason`.
/// Codes are matched loosely: case-insensitive, punctuation and spacing
/// ignored, so `TnS`, `T&S` and `trust and safety` all land on the same
/// clause. Free text (including Chinese) that matches no code is kept
/// verbatim as the label with no anchor.
fn classify_termination_reason(raw: &str) -> TerminationReason {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return TerminationReason { label: String::new(), anchor: None };
    }
    let norm: String = trimmed.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_uppercase();
    let (label, anchor): (&str, Option<&'static str>) = match norm.as_str() {
        "TNS" | "TS" | "TNSTS" | "TRUSTANDSAFETY" | "TRUSTSAFETY" => ("信任與安全（Trust & Safety）", Some("termination")),
        "PRIVATE" | "PRIV" | "PG" | "PRIVATEGROUP" => ("本服務不適用於私密（非公開）群組", Some("eligibility")),
        "ABUSE" | "MISUSE" => ("濫用本項目服務", Some("prohibited")),
        "RESOURCE" | "RES" => ("資源管理與服務完整性", Some("termination")),
        _ => (trimmed, None),
    };
    TerminationReason { label: label.to_string(), anchor }
}

/// The notice posted to a group as `/leave` terminates service for it.
/// Deliberately formal and self-contained: it's the only thing that group
/// will ever receive from the bot again, so it has to state the scope, the
/// grounds, and the one appeal route without assuming the reader can ask a
/// follow-up. Same `❖` heading and `@SEELE_01_BOT` appeal line as the
/// blacklist-reason notice, so it reads as the same project's voice.
fn service_termination_text(reason: &TerminationReason) -> String {
    // `使用規範` links to the terms as a whole; the specific ground only
    // appears when a maintainer typed one, and when it's a known code the
    // detail line links straight to the exact clause. `/leave` defaults its
    // stored reason to "違反使用規則", which rendered here would just restate
    // the sentence it hangs off, so the empty case adds no detail line.
    let label = reason.label.trim();
    let detail = if label.is_empty() {
        String::new()
    } else if let Some(anchor) = reason.anchor {
        format!("\n<b>具體事由</b>：<a href=\"{TERMS_URL}#{anchor}\">{}</a>", escape_html(label))
    } else {
        format!("\n<b>具體事由</b>：{}", escape_html(label))
    };
    format!(
        "<b>❖ 服務終止通知</b>\n\n\
         本項目 Spam Protection Bot（SPB）已對此群組終止服務，即刻生效。\n\n\
         <b>範圍</b>\n\
         • 此群組不得再使用本項目的任何服務與功能。\n\
         • 此群組的擁有者與管理團隊，亦不得以任何形式參與、使用、操作或存取本項目服務。\n\
         • 本禁令針對管理團隊整體生效，而非僅限特定帳號，並適用於該團隊目前及日後建立的其他群組。\n\n\
         <b>原因</b>\n\
         經審查，此群組的使用方式違反本項目的<a href=\"{TERMS_URL}\">使用規範</a>。\
         基於本項目資源管理與服務完整性之考量，我們採取此措施。{detail}\n\n\
         <b>紀錄</b>\n\
         此群組將被列入本項目的服務終止紀錄。如為維護本項目、使用者或公眾權益所必要，我們保留日後公開更多資訊的權利。\n\n\
         <b>申訴</b>\n\
         在有限情況下，可透過 @SEELE_01_BOT 提出上訴。此為唯一受理管道，其他方式一律不予處理。"
    )
}

/// Sent to a group the bot is pulled into while either the group itself or
/// whoever added it is on a project denial list. Much shorter than
/// `service_termination_text` on purpose - the full notice was already
/// delivered when service was terminated, and this is just the bot
/// declining to stay.
fn service_denied_text() -> String {
    "<b>❖ 服務終止通知</b>\n\n此群組或邀請本機器人的帳號已被本項目終止服務，機器人不會留在此群組。\n\n如有異議，請透過 @SEELE_01_BOT 提出上訴。".to_string()
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
    let reason = case.matched_rule_pattern.as_deref().unwrap_or("-");
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

/// Announces a netban catching someone *in this group* - they posted here,
/// or just joined. Deliberately not sent when the ban is first propagated
/// (see `propagate_network_ban`): a group only hears about a network ban
/// once the person it concerns actually turns up, rather than every time
/// someone is banned somewhere else.
///
/// Same self-delete window as `notify_group` - this is routine ambient
/// noise for group admins, not something that needs to stick around in the
/// chat permanently.
async fn notify_netban_sync(bot: &Bot, chat_id: ChatId, target_user_id: i64, case_id: &str) {
    let text = format!("<b>跨群組黑名單同步封禁</b>\n用戶 <code>{target_user_id}</code> 已因跨群組黑名單同步封禁。\n原始案例: <code>{case_id}</code>");
    let Ok(sent) = bot.send_message(chat_id, text).parse_mode(ParseMode::Html).await else { return };
    let bot = bot.clone();
    let message_id = sent.id;
    tokio::spawn(async move {
        sleep(Duration::from_secs(180)).await;
        let _ = bot.delete_message(chat_id, message_id).await;
    });
}

/// Same shape as `notify_netban_sync`, but for check_reban_and_act's
/// same-chat case - worth flagging to admins since a ban that let its
/// target back in once might do so again, unlike a routine netban sync.
async fn notify_reban_sync(bot: &Bot, chat_id: ChatId, target_user_id: i64, case_id: &str) {
    let text = format!("<b>已封禁用戶再次發言</b>\n用戶 <code>{target_user_id}</code> 在本群仍有生效中的封禁記錄，但成功再次發言，已刪除訊息並重新封禁。\n原始案例: <code>{case_id}</code>");
    let Ok(sent) = bot.send_message(chat_id, text).parse_mode(ParseMode::Html).await else { return };
    let bot = bot.clone();
    let message_id = sent.id;
    tokio::spawn(async move {
        sleep(Duration::from_secs(180)).await;
        let _ = bot.delete_message(chat_id, message_id).await;
    });
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
        let group_line = match chat_id {
            Some(id) => {
                let title = bot.get_chat(ChatId(id)).await.ok().and_then(|c| c.title().map(escape_html));
                match title {
                    Some(title) => format!("\n<b>群組</b>: {title} (<code>{id}</code>)"),
                    None => format!("\n<b>群組</b>: <code>{id}</code>"),
                }
            }
            None => String::new(),
        };
        let text = format!(
            "<b>維護操作 #{action_id}</b>\n<b>指令</b>: <code>{}</code>\n<b>操作者</b>: {} (<code>{actor_id}</code>){group_line}\n<b>內容</b>: {}\n{revert_hint}",
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
/// Why an `unbanChatMember` failure still leaves the user un-banned, if it
/// does.
///
/// The point of an unban is the *state* "this user is not banned here", not
/// the API call returning ok. Telegram rejects the call in several
/// situations where that state already holds, and treating those as failures
/// was actively harmful: an accepted appeal for a user who had since been
/// made an administrator reported back to PM as "appeal failed". Worse,
/// `reverse_ban_case` bailed on the first error, so the case was never
/// marked `Unbanned` — it stayed active, kept the user on the shared netban
/// list, and left the mistaken training sample in the model.
///
/// Returns `None` for errors that are genuine failures (missing rights, for
/// instance), which still have to be reported.
fn unban_noop_reason(err: &teloxide::RequestError) -> Option<&'static str> {
    // Telegram has no dedicated error variant for most of these, so they
    // arrive as `Unknown` with the raw description; match on that text.
    let text = err.to_string().to_lowercase();
    if text.contains("user is an administrator") {
        return Some("對象目前是該群管理員，本來就未被封禁");
    }
    if text.contains("chat not found") || text.contains("bot was kicked") || text.contains("bot is not a member") {
        return Some("機器人已不在該群組");
    }
    if text.contains("user not found") || text.contains("participant_id_invalid") || text.contains("user_not_participant") {
        return Some("對象不在該群組");
    }
    None
}

async fn reverse_ban_case(bot: &Bot, runtime: &Runtime, mut case: CaseRecord, actor_id: i64, actor_name: &str) -> Result<String, String> {
    let mut noop_note = String::new();
    if let Err(err) = bot.unban_chat_member(ChatId(case.chat_id), UserId(case.target_user_id as u64)).await {
        match unban_noop_reason(&err) {
            // Already not banned. Carry on with the rest of the reversal -
            // the case still has to be closed, the training sample removed,
            // and the netban entry cleared, none of which happens if we
            // return here.
            Some(reason) => noop_note = format!("（{reason}）"),
            None => return Err(format!("解封失敗：{err}")),
        }
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
    broadcast_unban_if_fully_clear(bot, runtime, case.target_user_id).await;

    let network_note = if network_targets.is_empty() {
        String::new()
    } else {
        format!("，並在 {} 個跨群組黑名單同步的群組中解封", network_targets.len())
    };
    Ok(format!("已解封用戶{noop_note}，並撤銷 case <code>{}</code>、移除 {removed} 筆對應訓練樣本{network_note}。", case.id))
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

/// Keeps PM's local ban-status cache (`bad_ids["users"]`) in sync so a
/// banned user's very first cold DM is routed straight to the self-appeal
/// flow instead of normal ticket handling - without this, PM only learns
/// about a ban via a live `/status` check, which defeats the point of
/// "just message the bot to appeal." No-ops if no exchange channel is
/// configured. Unconditional, unlike `propagate_network_ban` below - PM's
/// appeal flow has nothing to do with which groups opted into netban.
async fn broadcast_ban_status(bot: &Bot, runtime: &Runtime, user_id: i64, banned: bool) {
    let Some(chat) = runtime.exchange_channel().await else { return };
    // "report"/"bad" with an explicit is_banned field, matching bad_detail's
    // shape - not the old add/remove-verb encoding. request_id is genuinely
    // absent here (there's nothing to correlate an unsolicited push to),
    // unlike handle_exchange_query_bad's reply below.
    send_exchange_message(bot, chat, "report", "bad", serde_json::json!({ "id": user_id, "is_banned": banned })).await;
}

/// A ban lifted in just one group doesn't mean the user is clear
/// everywhere - they may have an independent active ban in another,
/// unrelated group. Only announce "not banned" once none remain anywhere;
/// otherwise stay silent rather than falsely clearing PM's cache.
async fn broadcast_unban_if_fully_clear(bot: &Bot, runtime: &Runtime, user_id: i64) {
    let still_banned = runtime.find_active_bans_for_user(user_id).await.map(|cases| !cases.is_empty()).unwrap_or(true);
    if !still_banned {
        broadcast_ban_status(bot, runtime, user_id, false).await;
    }
}

/// Whether a ban belongs on the shared project blacklist. Pulled out of
/// `propagate_network_ban` so the rule itself is directly testable - it's
/// the boundary that decides what one group can impose on every other, so
/// it deserves to be pinned down rather than only exercised through a live
/// Telegram call.
///
/// The dividing line is whether the *project* decided, or one group did:
///
/// - `ReportApproved` - a maintainer reviewed it in the report channel and
///   pressed approve. The strongest signal there is.
/// - `GuestBotBan` / `GuestInvokerBan` - guest-mode abuse is inherently
///   cross-group (the whole exploit is posting into groups the bot was
///   never added to), and the target is a throwaway spam account.
/// - `AutoBan` - only if it carries a score clearing the **global**
///   threshold. That covers the model and maintainer-managed regex rules
///   (which score 1.0). It deliberately excludes the module checks
///   (NoHalal, NoLongName): those are unscored, and they're per-group
///   *policy* opt-ins rather than spam determinations - a group choosing to
///   ban Arabic script or long names is making a house rule, not finding
///   something everyone else should ban too.
/// - `SpamBan` (`/sb`) - excluded. It's one group admin's judgment about
///   their own room; there's no project review behind it, so it must not
///   become a project-wide ban. Send it through `/spam` if it deserves one.
///
/// Note the origin group's netban setting plays no part: it governs what a
/// group *receives*, never what it can impose on everyone else.
fn netban_eligible(action: &ActionKind, model_score: Option<f64>, global_threshold: f64, reason: Option<&str>) -> bool {
    // Pure bot-mention spam qualifies however it was caught, including a
    // manual /sb - a bare bot handle is an unambiguous spam-bot advert, and
    // the same /sb also creates a regex rule, so the account belongs on the
    // shared blacklist too.
    if reason == Some("BOTSPAM") {
        return true;
    }
    match action {
        ActionKind::ReportApproved | ActionKind::GuestBotBan | ActionKind::GuestInvokerBan => true,
        ActionKind::AutoBan => model_score.is_some_and(|score| score >= global_threshold),
        _ => false,
    }
}

/// Decides whether a ban belongs on the shared project blacklist, and if so
/// records that on the case and pushes it to every group receiving netban.
/// The single chokepoint every ban path already calls.
///
/// Contributing to the blacklist is *not* gated on the origin group having
/// netban switched on - the blacklist is a project-level record, so a
/// message that clears the project's own spam bar belongs on it wherever it
/// was caught. Receiving stays opt-in: only groups with netban enabled ever
/// get a propagated ban (here) or enforce one (`check_netban_and_act`).
///
/// Posts a `/sb`'s evidence to the report channel for a maintainer to
/// accept or discard as training data. The ban itself already happened and
/// isn't in question here - this only decides whether the text is allowed
/// to move the shared model, which used to happen automatically the moment
/// any group admin typed `/sb`.
///
/// Skipped when there's nothing a human could usefully label: an empty or
/// token-less message (a sticker, a photo with no caption) would train on
/// nothing, so it would only be noise in the review queue.
async fn queue_training_review(bot: &Bot, runtime: &Runtime, case: &CaseRecord) {
    if is_empty_ml_text(&case.evidence_text) {
        return;
    }
    let body = format!(
        "<b>待審核訓練樣本</b>（來自 /sb）\n\n<b>對象</b>: {} (<code>{}</code>)\n<b>操作者</b>: {}\n<b>群組</b>: <code>{}</code>\n<b>內容</b>: <blockquote>{}</blockquote>\n<b>案例</b>: <code>{}</code>\n\n批准後才會寫入模型；拒絕則只保留封禁、不影響模型。",
        escape_html(&case.target_name),
        case.target_user_id,
        escape_html(case.actor_name.as_deref().unwrap_or("unknown")),
        case.chat_id,
        escape_html(&case.evidence_text),
        case.id,
    );
    let buttons = InlineKeyboardMarkup::new(vec![vec![
        InlineKeyboardButton::callback("批准訓練", format!("train:approve:{}", case.id)),
        InlineKeyboardButton::callback("拒絕訓練", format!("train:reject:{}", case.id)),
    ]]);
    let _ = bot
        .send_message(ChatId(runtime.config.report_channel_id), body)
        .parse_mode(ParseMode::Html)
        .reply_markup(buttons)
        .await;
}

/// A scored ban is judged against the **global** threshold, never the origin
/// group's own - a group running a lowered `spam_threshold_override` bans at
/// its own bar locally, but can't push those below-bar bans onto everyone
/// else. See `netban_eligible` for which actions qualify at all.
async fn propagate_network_ban(bot: &Bot, runtime: &Runtime, case: &CaseRecord) {
    let global = runtime.current_threshold().await.unwrap_or(runtime.config.spam_threshold);
    if !netban_eligible(&case.action, case.model_score, global, case.matched_rule_pattern.as_deref()) {
        return;
    }
    commit_network_ban(bot, runtime, case).await;
}

/// Adds a case to the shared blacklist and bans its target in every
/// subscribed (netban-enabled) group. Eligibility is the caller's call -
/// `propagate_network_ban` gates on `netban_eligible`, while a reviewer
/// approving a /sb promotes that ban here directly (one admin's /sb stays
/// local; a reviewer approving it is the second gate that makes it
/// project-wide).
async fn commit_network_ban(bot: &Bot, runtime: &Runtime, case: &CaseRecord) {
    if runtime.is_global_whitelisted(case.target_user_id).await.unwrap_or(false) {
        return;
    }
    let _ = runtime.mark_netban_eligible(&case.id).await;

    let targets = runtime.list_netban_enabled_chats().await.unwrap_or_default();
    for chat_id in targets {
        if chat_id == case.chat_id {
            continue;
        }
        if runtime.is_group_whitelisted(chat_id, case.target_user_id).await.unwrap_or(false) {
            continue;
        }
        if bot.ban_chat_member(ChatId(chat_id), UserId(case.target_user_id as u64)).await.is_ok() {
            // Silent on purpose. Propagation hits every receiving group at
            // once, almost always about someone who has never posted there -
            // announcing it just fills unrelated groups with notices about
            // strangers. The record is still written, and the groups that
            // actually meet this user get told at that point instead, by
            // check_netban_and_act / the join-time catch-up in
            // notify_bot_added.
            let _ = runtime.record_network_ban_target(&case.id, chat_id).await;
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

/// A `reply_to_message()` that's an actual reply the user made - not the
/// synthetic reply Telegram automatically attaches to every message sent
/// inside a forum topic, which points at that topic's own creation/service
/// message. That message's `from` is whoever *created the topic*, not
/// whoever's currently posting, and it carries no real text - so treating
/// it as a genuine reply would silently target/train on the topic creator
/// for a bare command nobody actually replied with. Every place in this
/// file that resolves a command's target or content from a reply must go
/// through this, not raw `.reply_to_message()`.
fn real_reply(message: &Message) -> Option<&Message> {
    let reply = message.reply_to_message()?;
    if reply.forum_topic_created().is_some() {
        return None;
    }
    Some(reply)
}

async fn extract_reply_context(message: &Message) -> Option<(i64, String, i32, String)> {
    let reply = real_reply(message)?;
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

    if runtime.is_maintainer(user_id).await || is_platform_pseudo_user(user_id) {
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

    if runtime.is_maintainer(user_id).await || is_platform_pseudo_user(user_id) {
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
    notify_netban_sync(bot, message.chat.id, user_id, &prior_case.id).await;
    true
}

/// Same-chat ban-evasion safety net: a Telegram ban is supposed to make
/// re-posting in that exact chat impossible, so this should normally never
/// trip - but if it somehow does (a ban that silently failed to take, a
/// Telegram-side enforcement gap, anything else outside this bot's control),
/// this catches the repost instead of letting it fall through to ordinary
/// scoring, which might not flag it at all if the content itself looks
/// innocuous. Deliberately has nothing to do with netban - that flag only
/// controls whether a ban *propagates to other chats*, not whether this
/// chat remembers its own past bans, so unlike check_netban_and_act this
/// always applies regardless of any module setting.
async fn check_reban_and_act(bot: &Bot, runtime: &Arc<Runtime>, message: &Message) -> bool {
    if !message.chat.is_group() && !message.chat.is_supergroup() {
        return false;
    }
    let Some(user) = message.from.as_ref() else { return false; };
    if user.is_bot {
        return false;
    }
    let chat_id = message.chat.id.0;
    let user_id = user.id.0 as i64;

    if runtime.is_maintainer(user_id).await || is_platform_pseudo_user(user_id) {
        return false;
    }
    if runtime.is_global_whitelisted(user_id).await.unwrap_or(false) {
        return false;
    }
    if runtime.is_group_whitelisted(chat_id, user_id).await.unwrap_or(false) {
        return false;
    }

    let Ok(Some(prior_case)) = runtime.find_active_ban_in_chat(chat_id, user_id).await else {
        return false;
    };

    let _ = bot.delete_message(message.chat.id, message.id).await;
    let _ = bot.ban_chat_member(message.chat.id, user.id).await;
    notify_reban_sync(bot, message.chat.id, user_id, &prior_case.id).await;
    true
}

/// Rejected `/spam` reports a user may accumulate before losing the
/// command. Cleared by a maintainer with `/report_reset`.
const REPORT_STRIKE_LIMIT: i64 = 3;

/// Filename extensions that execute code, or install something that does,
/// on at least one common platform. Archives are deliberately absent: a
/// .zip is ordinary enough that banning on it would cost more than it saves,
/// even though one can carry any of these inside.
const EXECUTABLE_EXTENSIONS: &[&str] = &[
    "exe", "scr", "com", "pif", "bat", "cmd", "vbs", "vbe", "js", "jse", "wsf", "wsh", "hta",
    "msi", "msp", "cpl", "reg", "lnk", "ps1", "psm1", "jar", "apk", "app", "dmg", "pkg", "deb",
    "rpm", "sh", "bash", "run", "bin", "elf", "dll", "sys", "iso", "img",
];

/// The extension of `name`, lowercased, ignoring any Telegram-style trailing
/// dot tricks. Returns `None` when there is nothing after the final dot.
fn file_extension(name: &str) -> Option<String> {
    let trimmed = name.trim().trim_end_matches('.');
    let (_, ext) = trimmed.rsplit_once('.')?;
    if ext.is_empty() || ext.chars().any(|c| !c.is_ascii_alphanumeric()) {
        return None;
    }
    Some(ext.to_lowercase())
}

/// Which attachment-policy module, if any, this message trips. Split out
/// from the handler so the classification is testable without a live
/// Telegram message round trip.
fn attachment_violation(message: &Message, settings: &GroupModuleSettings) -> Option<&'static str> {
    if settings.no_contact && message.contact().is_some() {
        return Some("CONTACT");
    }
    if settings.no_voice && message.voice().is_some() {
        return Some("VOICE");
    }
    if settings.no_exec {
        if let Some(doc) = message.document() {
            // Filename first: Telegram's mime_type is client-supplied and a
            // sender controls it, so it is corroboration rather than proof.
            let by_name = doc
                .file_name
                .as_deref()
                .and_then(file_extension)
                .is_some_and(|ext| EXECUTABLE_EXTENSIONS.contains(&ext.as_str()));
            let by_mime = doc.mime_type.as_ref().is_some_and(|m| {
                let m = m.essence_str();
                m == "application/x-msdownload"
                    || m == "application/x-msdos-program"
                    || m == "application/vnd.microsoft.portable-executable"
                    || m == "application/x-executable"
                    || m == "application/vnd.android.package-archive"
            });
            if by_name || by_mime {
                return Some("EXEC_FILE");
            }
        }
    }
    None
}

/// Bans on what a message *carries* rather than what it says: a shared
/// contact card, a voice recording, or an executable attachment. Each is
/// its own opt-in module, all default off.
///
/// These run as their own check rather than inside `auto_moderate` because
/// that one only sees text and captions - a contact card or a bare .exe has
/// neither, so it scores as empty and is skipped entirely. They are also the
/// only checks here that fire with no text to judge, which is exactly why
/// they are opt-in: the group is asserting a policy, not reporting spam.
async fn check_attachment_policy_and_act(bot: &Bot, runtime: &Arc<Runtime>, message: &Message) -> bool {
    if !message.chat.is_group() && !message.chat.is_supergroup() {
        return false;
    }
    let Some(user) = message.from.as_ref() else { return false; };
    if user.is_bot {
        return false;
    }
    let chat_id = message.chat.id.0;
    let user_id = user.id.0 as i64;

    if runtime.is_maintainer(user_id).await || is_platform_pseudo_user(user_id) {
        return false;
    }

    let settings = runtime.get_group_modules(chat_id).await.unwrap_or_default();
    let Some(reason) = attachment_violation(message, &settings) else { return false; };

    if runtime.is_global_whitelisted(user_id).await.unwrap_or(false) {
        return false;
    }
    if runtime.is_group_whitelisted(chat_id, user_id).await.unwrap_or(false) {
        return false;
    }
    if is_group_admin(bot, message.chat.id, user_id).await {
        return false;
    }

    let _ = bot.delete_message(message.chat.id, message.id).await;
    let _ = ban_user(bot, message.chat.id, user_id).await;

    let case = CaseRecord {
        id: Uuid::new_v4().to_string(),
        action: ActionKind::AutoBan,
        chat_id,
        target_user_id: user_id,
        target_name: short_user(user),
        actor_user_id: None,
        actor_name: None,
        source_message_id: Some(message.id.0),
        evidence_text: extract_full_text(message),
        model_score: None,
        matched_rule_id: None,
        matched_rule_pattern: Some(reason.to_string()),
        status: "auto_banned".to_string(),
        log_message_id: None,
        created_at: Utc::now(),
    };
    let log_message_id = log_action(bot, runtime, &case).await.unwrap_or_default();
    let mut updated = case.clone();
    updated.log_message_id = Some(log_message_id);
    let _ = store_case(runtime, &updated).await;
    let _ = notify_group(bot, runtime, &updated, log_message_id, "<b>附件政策封禁</b>").await;
    // Unscored AutoBan, so netban_eligible() rejects it - a group's own
    // attachment policy is not a project-wide spam finding.
    propagate_network_ban(bot, runtime, &updated).await;
    broadcast_ban_status(bot, runtime, updated.target_user_id, true).await;
    true
}

/// Telegram's "guest mode" (https://core.telegram.org/api/bots/guest-mode)
/// lets any user @-mention a bot into posting directly into a group that bot
/// was never added to. Bot API delivers the resulting message like any
/// other - `from` is the guest bot's own account - but with nothing linking
/// it back to whoever actually invoked it. Every other message-time check in
/// this file (check_flood_and_act, check_netban_and_act, auto_moderate,
/// score_only) deliberately skips `is_bot` messages, so it never moderates a
/// legitimately-added utility bot - which is exactly what let guest-mode
/// posts slip through completely unchecked. The only reliable way to tell a
/// guest-mode post apart from a real member bot's message is chat
/// membership: a properly-added bot comes back "member"/"administrator"/
/// "restricted"; a guest-mode poster was never added at all, so it comes
/// back "left" - or the lookup just fails outright. A lookup error is
/// treated as "don't know" and skipped, never as "assume guest", since a
/// transient API hiccup must never be grounds for banning a real member bot.
async fn check_guest_bot_and_act(bot: &Bot, runtime: &Arc<Runtime>, message: &Message) -> bool {
    if !message.chat.is_group() && !message.chat.is_supergroup() {
        return false;
    }
    let Some(user) = message.from.as_ref() else { return false; };
    if !user.is_bot {
        return false;
    }
    let chat_id = message.chat.id.0;
    let user_id = user.id.0 as i64;

    // Telegram's own pseudo-accounts look *exactly* like a guest-mode bot:
    // GroupAnonymousBot (an admin posting anonymously) and the channel
    // sender are flagged `is_bot` and are genuinely not in the member list,
    // so the membership test below says "Left" for both. Banning them hits
    // whichever real admin was hiding behind them and deletes their message.
    // Every other check skips these by never touching bots at all; this
    // function only ever runs on bots, so it needs the guard explicitly.
    if is_platform_pseudo_user(user_id) {
        return false;
    }

    let settings = runtime.get_group_modules(chat_id).await.unwrap_or_default();
    if !settings.guest_ban {
        return false;
    }

    // Reuses the existing whitelist commands as an escape hatch: a
    // maintainer/admin who wants to keep some other legitimately-invited
    // guest bot around can just /white its account id.
    if runtime.is_global_whitelisted(user_id).await.unwrap_or(false) {
        return false;
    }
    if runtime.is_group_whitelisted(chat_id, user_id).await.unwrap_or(false) {
        return false;
    }
    if runtime.me_id(bot).await == Some(user.id) {
        return false;
    }

    let Ok(member) = bot.get_chat_member(message.chat.id, user.id).await else { return false; };
    if !matches!(member.kind, teloxide::types::ChatMemberKind::Left | teloxide::types::ChatMemberKind::Banned(_)) {
        return false;
    }

    let _ = bot.delete_message(message.chat.id, message.id).await;
    let _ = bot.ban_chat_member(message.chat.id, user.id).await;

    let case = CaseRecord {
        id: Uuid::new_v4().to_string(),
        action: ActionKind::GuestBotBan,
        chat_id,
        target_user_id: user_id,
        target_name: short_user(user),
        actor_user_id: None,
        actor_name: None,
        source_message_id: Some(message.id.0),
        evidence_text: extract_full_text(message),
        model_score: None,
        matched_rule_id: None,
        matched_rule_pattern: Some("GUEST_MODE".to_string()),
        status: "guest_bot_banned".to_string(),
        log_message_id: None,
        created_at: Utc::now(),
    };
    let log_message_id = log_action(bot, runtime, &case).await.unwrap_or_default();
    let mut updated = case.clone();
    updated.log_message_id = Some(log_message_id);
    let _ = store_case(runtime, &updated).await;
    let _ = notify_group(bot, runtime, &updated, log_message_id, "<b>訪客模式機器人已封鎖</b>").await;
    propagate_network_ban(bot, runtime, &updated).await;
    broadcast_ban_status(bot, runtime, updated.target_user_id, true).await;

    // The guest bot's own account is only half the problem - whoever typed
    // `@thatbot` to summon it is a real member who chose to invite spam into
    // the chat, and their message (plus their spammy profile, per the
    // report that led to this) stays behind untouched otherwise. Bot API
    // gives no way to identify them directly (see this function's doc
    // comment), so this falls back to the best available signal: the most
    // recent bare `@thatbot` mention in this chat. See
    // find_recent_guest_invoker for exactly what counts as a match.
    if let Some(bot_username) = user.username.as_deref() {
        if let Some((invoker_id, invoker_msg_id, invoker_name, invoker_text)) = runtime.find_recent_guest_invoker(chat_id, bot_username).await {
            // One summon can pull in several guest bots, so this runs once
            // per guest reply for the same invoking message. Forget it
            // immediately and skip anyone already banned here, so the
            // follow-up replies don't each open a duplicate case.
            runtime.forget_recent_message(chat_id, invoker_msg_id).await;
            let invoker_exempt = runtime.is_maintainer(invoker_id).await
                || is_platform_pseudo_user(invoker_id)
                || runtime.is_global_whitelisted(invoker_id).await.unwrap_or(false)
                || runtime.is_group_whitelisted(chat_id, invoker_id).await.unwrap_or(false)
                || runtime.find_active_ban_in_chat(chat_id, invoker_id).await.ok().flatten().is_some()
                || is_group_admin(bot, message.chat.id, invoker_id).await;
            if !invoker_exempt {
                let _ = bot.delete_message(message.chat.id, invoker_msg_id).await;
                let _ = bot.ban_chat_member(message.chat.id, UserId(invoker_id as u64)).await;

                let invoker_case = CaseRecord {
                    id: Uuid::new_v4().to_string(),
                    action: ActionKind::GuestInvokerBan,
                    chat_id,
                    target_user_id: invoker_id,
                    target_name: invoker_name,
                    actor_user_id: None,
                    actor_name: None,
                    source_message_id: Some(invoker_msg_id.0),
                    evidence_text: invoker_text,
                    model_score: None,
                    matched_rule_id: None,
                    matched_rule_pattern: Some("GUEST_MODE_INVOKER".to_string()),
                    status: "guest_invoker_banned".to_string(),
                    log_message_id: None,
                    created_at: Utc::now(),
                };
                let invoker_log_id = log_action(bot, runtime, &invoker_case).await.unwrap_or_default();
                let mut invoker_updated = invoker_case.clone();
                invoker_updated.log_message_id = Some(invoker_log_id);
                let _ = store_case(runtime, &invoker_updated).await;
                let _ = notify_group(bot, runtime, &invoker_updated, invoker_log_id, "<b>訪客模式召喚者已封鎖</b>").await;
                propagate_network_ban(bot, runtime, &invoker_updated).await;
                broadcast_ban_status(bot, runtime, invoker_updated.target_user_id, true).await;
            }
        }
    }

    true
}

async fn handle_command(bot: Bot, runtime: Arc<Runtime>, message: Message) -> ResponseResult<()> {
    let Some(text) = message.text() else { return Ok(()); };
    let Some(from) = message.from.as_ref() else { return Ok(()); };
    let from_id = from.id.0 as i64;
    // The host console's trigger is an env secret (HOSTCTL_CMD, default
    // "/hostctl"), resolved here rather than in the static parse_command so
    // the actual command string never has to appear in source. Only the host
    // can reach it at all (is_host), so this override is invisible to anyone
    // else even if they typed the same word.
    let cmd = {
        let first = text.split_whitespace().next().unwrap_or("");
        let first = first.split('@').next().unwrap_or(first);
        if is_host(from_id) && first.eq_ignore_ascii_case(&runtime.config.hostctl_cmd) {
            ModerationCommand::HostCtl(text.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim().to_string())
        } else {
            parse_command(text)
        }
    };
    let is_private_maintainer = message.chat.is_private() && runtime.is_maintainer(from_id).await;

    // Project-level denial: someone on the `/forbid` list gets no response
    // to anything, anywhere. Silent rather than an error reply - there's no
    // appeal to conduct here (that's @SEELE_01_BOT's job) and answering
    // would just invite argument in whatever group they tried it in.
    // Maintainers are exempt so a mis-`/forbid` can never lock out the
    // people who'd have to undo it.
    if !runtime.is_maintainer(from_id).await && runtime.is_user_banned(from_id).await {
        return Ok(());
    }

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
        ModerationCommand::HostCtl(arg) => {
            // Host-only admin console. Gated on is_host - Telegram
            // authenticates from_id, so it's unforgeable. Anyone else gets no
            // response, and it's not listed in /help.
            if !is_host(from_id) {
                return Ok(());
            }
            let mut arg = arg.as_str();
            // Second factor: if a secret is set (DB, or the HOSTCTL_SECRET env
            // on first run), the first token must be that secret or the whole
            // command is silently ignored - no reply, no hint it exists. This
            // is what protects the console if the host account is ever
            // compromised or left open, and stays effective once the source is
            // closed. Change it with the PW op; recover it via SQL if forgotten.
            let secret = runtime.hostctl_secret().await;
            if let Some(secret) = secret.as_deref() {
                match arg.split_once(char::is_whitespace) {
                    Some((first, tail)) if first == secret => arg = tail.trim_start(),
                    // Also allow the secret alone (e.g. to get the ?? legend).
                    None if arg == secret => arg = "",
                    _ => return Ok(()),
                }
            }
            // GDS-style grammar: a two-letter op, then optional `.<id>` and/or
            // `/<text>` params. Deliberately terse and opaque - meaningless to
            // anyone glancing at it in a group without the manual. Examples:
            //   MP            pin the replied message
            //   AP.123456789  promote user 123456789 to admin
            //   AR            demote the replied user
            //   UB.123        ban user 123    UR.123  unban
            //   CT/New Name   set chat title
            //   BS/hello      make the bot say "hello" here
            //   BX.-100123/hi send "hi" to chat -100123
            //   IL            invite link     ??  op legend
            let s = arg.trim();
            let chat_id = message.chat.id;
            let replied_id = real_reply(&message).map(|m| m.id);
            let reply_uid = real_reply(&message).and_then(|m| m.from.as_ref()).map(|u| u.id.0 as i64);

            fn report(r: Result<(), teloxide::RequestError>) -> String {
                match r {
                    Ok(()) => "OK".to_string(),
                    Err(e) => format!("ERR {e}"),
                }
            }

            let two = s.len() >= 2 && s.is_char_boundary(2);
            let op = if two { s[..2].to_uppercase() } else { s.to_uppercase() };
            let mut tail = if two { &s[2..] } else { "" };
            // Optional `@<chat_id>` selector: lets a chat-scoped op act on
            // another group instead of the one the command was typed in.
            let mut chat_sel: Option<i64> = None;
            if let Some(r) = tail.strip_prefix('@') {
                let end = r.find(['.', '/']).unwrap_or(r.len());
                chat_sel = r[..end].trim().parse().ok();
                tail = &r[end..];
            }
            let (id_seg, text_seg) = match tail.find('/') {
                Some(i) => (&tail[..i], Some(tail[i + 1..].trim())),
                None => (tail, None),
            };
            let dot_id: Option<i64> = id_seg.trim().trim_start_matches('.').trim().parse().ok();
            let text = text_seg.unwrap_or("");
            // Which chat an op acts on. Chat-scoped ops (IL/CT/BS) take the
            // target from @chat, or a bare `.id`, else the current chat. User
            // ops (AP/AR/UB/UR) take the chat from @chat only (bare `.id` is
            // the user there), else the current chat.
            let chat_scoped = chat_sel.or(dot_id).map(ChatId).unwrap_or(chat_id);
            let chat_user = chat_sel.map(ChatId).unwrap_or(chat_id);
            let target_id: Option<i64> = dot_id.or(reply_uid);

            let out: String = match op.as_str() {
                "" | "??" => (
                    "<b>主機終端 指令表</b>\n\
                     <code>MP</code>　置頂（回覆訊息）\n\
                     <code>MU</code>　取消置頂（回覆；無回覆＝全部）\n\
                     <code>MX</code>　刪除（回覆訊息）\n\
                     <code>IL</code>[<code>@群組</code>]　產生邀請連結\n\
                     <code>AP</code>[<code>@群組</code>]<code>.id</code>　設為管理員（或回覆）\n\
                     <code>AR</code>[<code>@群組</code>]<code>.id</code>　取消管理員（或回覆）\n\
                     <code>UB</code>[<code>@群組</code>]<code>.id</code>　封鎖（或回覆）\n\
                     <code>UR</code>[<code>@群組</code>]<code>.id</code>　解除封鎖（或回覆）\n\
                     <code>CT</code>[<code>@群組</code>]<code>/文字</code>　改群名\n\
                     <code>BS</code>[<code>@群組</code>]<code>/文字</code>　以機器人發言\n\
                     <code>PW</code><code>/新密碼</code>　變更密碼\n\
                     <i>不加 @群組 時對目前所在的群組操作。</i>\n\
                     <i>例：IL@-1002680968271 ＝ 產生該群邀請連結</i>"
                ).to_string(),
                "MP" => match replied_id {
                    Some(mid) => report(bot.pin_chat_message(chat_id, mid).await.map(|_| ())),
                    None => "ERR reply".to_string(),
                },
                "MU" => match replied_id {
                    Some(mid) => report(bot.unpin_chat_message(chat_id).message_id(mid).await.map(|_| ())),
                    None => report(bot.unpin_all_chat_messages(chat_id).await.map(|_| ())),
                },
                "MX" => match replied_id {
                    Some(mid) => report(bot.delete_message(chat_id, mid).await.map(|_| ())),
                    None => "ERR reply".to_string(),
                },
                "IL" => match bot.create_chat_invite_link(chat_scoped).await {
                    Ok(link) => link.invite_link,
                    Err(e) => format!("ERR {e}"),
                },
                "AP" => match target_id {
                    Some(uid) => report(
                        bot.promote_chat_member(chat_user, UserId(uid as u64))
                            .can_manage_chat(true)
                            .can_delete_messages(true)
                            .can_restrict_members(true)
                            .can_pin_messages(true)
                            .can_invite_users(true)
                            .can_manage_video_chats(true)
                            .can_change_info(true)
                            .await
                            .map(|_| ()),
                    ),
                    None => "ERR .id".to_string(),
                },
                "AR" => match target_id {
                    Some(uid) => report(
                        bot.promote_chat_member(chat_user, UserId(uid as u64))
                            .can_manage_chat(false)
                            .can_delete_messages(false)
                            .can_restrict_members(false)
                            .can_pin_messages(false)
                            .can_invite_users(false)
                            .can_manage_video_chats(false)
                            .can_change_info(false)
                            .can_promote_members(false)
                            .await
                            .map(|_| ()),
                    ),
                    None => "ERR .id".to_string(),
                },
                "UB" => match target_id {
                    Some(uid) => report(bot.ban_chat_member(chat_user, UserId(uid as u64)).await.map(|_| ())),
                    None => "ERR .id".to_string(),
                },
                "UR" => match target_id {
                    Some(uid) => report(bot.unban_chat_member(chat_user, UserId(uid as u64)).await.map(|_| ())),
                    None => "ERR .id".to_string(),
                },
                "CT" => {
                    if text.is_empty() {
                        "ERR /t".to_string()
                    } else {
                        report(bot.set_chat_title(chat_scoped, text).await.map(|_| ()))
                    }
                }
                "BS" => {
                    if text.is_empty() {
                        "ERR /t".to_string()
                    } else {
                        match bot.send_message(chat_scoped, text).parse_mode(ParseMode::Html).await {
                            Ok(_) => "OK".to_string(),
                            Err(e) => format!("ERR {e}"),
                        }
                    }
                }
                // Change the console secret. Reachable only after passing the
                // current secret gate above, so it can't be used to reset a
                // secret you don't already know.
                "PW" => {
                    if text.is_empty() {
                        "ERR /t".to_string()
                    } else {
                        match runtime.set_hostctl_secret(text).await {
                            Ok(()) => "OK 密碼已更新".to_string(),
                            Err(e) => format!("ERR {e}"),
                        }
                    }
                }
                _ => "ERR".to_string(),
            };
            // Keep it low-profile: remove the command that was typed, and let
            // the reply self-delete after a minute (long enough to read an
            // invite link, short enough not to linger for others).
            let sent = bot.send_message(chat_id, out).parse_mode(ParseMode::Html).await;
            let _ = bot.delete_message(chat_id, message.id).await;
            if let Ok(sent) = sent {
                let bot = bot.clone();
                tokio::spawn(async move {
                    sleep(Duration::from_secs(60)).await;
                    let _ = bot.delete_message(chat_id, sent.id).await;
                });
            }
        }
        ModerationCommand::Start | ModerationCommand::Help => {
            // Both, not just /start: they send the same text, so the button
            // appearing on one and not the other would just look like a bug.
            bot.send_message(message.chat.id, help_text())
                .parse_mode(ParseMode::Html)
                .reply_markup(terms_button("閱讀使用規範 / Terms of Use"))
                .await?;
        }
        ModerationCommand::HelpOp(section) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let mut text = help_op_text(&section);
            // The host console is documented only for the host, on the index
            // page. Other maintainers never see it listed.
            if is_host(from_id) && section.trim().is_empty() {
                text.push_str(
                    "\n\n<b>━━ host ━━</b>\n\
                     主持人終端：<code>/hostctl &lt;密碼&gt; ??</code> 查代碼表（GDS 語法，僅你可用，指令與回覆自動刪除）",
                );
            }
            bot.send_message(message.chat.id, text).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::MyId => {
            let requester = message.from.as_ref();
            let target_user = real_reply(&message)
                .and_then(|m| m.from.as_ref())
                .or(requester);
            let uid = target_user.map(|u| u.id.0.to_string()).unwrap_or_else(|| "unknown".to_string());
            let target_name = target_user.map(short_user).unwrap_or_else(|| "unknown".to_string());
            // The role line is only shown to staff. /id is a public command,
            // so printing everyone's role would let any member probe who is a
            // maintainer just by replying to them.
            // from_id is the person who typed /id. Maintainer covers the host.
            let viewer_is_staff = runtime.is_maintainer(from_id).await;
            let role_line = if viewer_is_staff {
                let role = match target_user.map(|u| u.id.0 as i64) {
                    Some(id) if is_host(id) => "項目主持人",
                    Some(id) if runtime.is_maintainer(id).await => "維護組",
                    Some(id) if runtime.is_reviewer(id).await => "審核員",
                    _ => "一般用戶",
                };
                format!("\n• 身分: {role}")
            } else {
                String::new()
            };
            let body = format!("<b>查詢結果</b>\n• 對象: <code>{target_name}</code>\n• Telegram ID: <code>{uid}</code>{role_line}");
            bot.send_message(message.chat.id, body).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::MyChat => {
            bot.send_message(message.chat.id, format!("這個群的 Chat ID: <code>{}</code>", message.chat.id.0)).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::ScoreTest(text) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以使用 /ml_score。");
            let target_msg = real_reply(&message).unwrap_or(&message);
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
        ModerationCommand::SetExchangeChannel(chat_id) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以設定交換頻道。");
            // No current-chat convenience here, unlike /setchat/-set_audit_log
            // - this is a broadcast channel the bot posts into as an admin,
            // not somewhere you'd naturally run a command from inside.
            let Some(value) = chat_id.trim().parse::<i64>().ok() else {
                bot.send_message(message.chat.id, "請提供有效的 Chat ID，例如 /set_exchange_channel -1001234567890。").await?;
                return Ok(());
            };
            runtime.set_exchange_channel(value).await;
            bot.send_message(message.chat.id, format!("已設定 PM 申訴橋接交換頻道為 <code>{value}</code>。")).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::Magic(module, chat_id, action) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以使用此指令。");
            let module = module.trim().to_string();
            let usage = "用法：/magic <module> <chat_id> <allow|disallow>";
            let Some(target_chat_id) = chat_id.trim().parse::<i64>().ok() else {
                bot.send_message(message.chat.id, usage).await?;
                return Ok(());
            };
            let allow = match action.trim().to_lowercase().as_str() {
                "allow" => true,
                "disallow" => false,
                _ => {
                    bot.send_message(message.chat.id, usage).await?;
                    return Ok(());
                }
            };
            runtime.set_module_allowed(&module, target_chat_id, allow, Some(from_id)).await.ok();
            log_maintainer_action(
                &bot, &runtime, from_id, &short_user(from), Some(target_chat_id), "/magic",
                &format!("module={module} chat_id={target_chat_id} allow={allow}"),
                UndoData::NotRevertible,
            ).await;
            bot.send_message(message.chat.id, format!("已將群組 <code>{target_chat_id}</code> 的 <code>{module}</code> 存取設為 {}。", if allow { "允許" } else { "不允許" })).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::Leave(reason) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以使用 /leave。");
            let (target_chat_id, reason_raw) = parse_leave_args(&reason);
            // A short code (TnS, private, …) resolves to a labelled clause and
            // links the notice straight to that ToU section; anything else is
            // carried through as free text. The blacklist row and audit entry
            // still want something written down, so fall back to a placeholder.
            let reason = classify_termination_reason(&reason_raw);
            let stored_reason = if reason.label.trim().is_empty() { "違反使用規則".to_string() } else { reason.label.clone() };
            let target_chat_id = target_chat_id.unwrap_or(message.chat.id.0);
            // Guard against locking the project out of its own plumbing.
            // A blacklisted chat is refused *before* command handling, so
            // banning one of these would leave nowhere to type /forgive
            // except a DM - and a bare /leave in a DM would blacklist the
            // maintainer's own chat id, which is worse still.
            let mut protected = vec![runtime.config.log_channel_id, runtime.config.report_channel_id];
            protected.extend(runtime.project_chat().await);
            protected.extend(runtime.audit_log_chat().await);
            protected.extend(runtime.exchange_channel().await);
            protected.extend(runtime.config.test_group_id);
            if target_chat_id >= 0 || protected.contains(&target_chat_id) {
                bot.send_message(message.chat.id, "只能對一般群組使用 /leave，不能用於私訊或項目自身的頻道/群組。").await?;
                return Ok(());
            }
            let project_chat = match runtime.project_chat().await {
                Some(id) => id,
                None => {
                    bot.send_message(message.chat.id, "尚未設定項目交流群，請先使用 /setchat。") .await?;
                    return Ok(());
                }
            };
            let button = InlineKeyboardMarkup::new(vec![vec![InlineKeyboardButton::url("前往項目交流群查詢", Url::parse(&project_chat_link(project_chat)).unwrap())]]);
            let _ = bot
                .send_message(ChatId(target_chat_id), service_termination_text(&reason))
                .parse_mode(ParseMode::Html)
                .reply_markup(button)
                .await;
            // Blacklist before leaving, so a re-add during the gap is still
            // refused by the guard in notify_bot_added.
            let _ = runtime.set_group_banned(target_chat_id, true, &stored_reason, Some(from_id)).await;
            let _ = bot.leave_chat(ChatId(target_chat_id)).await;
            log_maintainer_action(
                &bot,
                &runtime,
                from_id,
                &short_user(from),
                Some(target_chat_id),
                "/leave",
                &format!("終止服務並列入封禁：{stored_reason}"),
                UndoData::GroupBanned { chat_id: target_chat_id },
            )
            .await;
            bot.send_message(message.chat.id, format!("已終止對 <code>{target_chat_id}</code> 的服務並列入封禁群組。解除請用 <code>/forgive {target_chat_id}</code>。")).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::Forbid(args) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以使用 /forbid。");
            // Same shape as /leave's args: an id, then a free-text reason.
            // A reply works too, so a maintainer can act straight off a
            // message without copying ids around.
            let (target_id, reason) = parse_leave_args(&args);
            let target_id = target_id.or_else(|| real_reply(&message).and_then(|r| r.from.as_ref()).map(|u| u.id.0 as i64));
            let Some(target_id) = target_id else {
                bot.send_message(message.chat.id, "請提供 user_id 或回覆該用戶，例如 /forbid 12345 濫用服務。").await?;
                return Ok(());
            };
            if target_id <= 0 {
                bot.send_message(message.chat.id, "/forbid 只接受 user_id。要封禁群組請用 /leave。").await?;
                return Ok(());
            }
            if runtime.is_maintainer(target_id).await || is_platform_pseudo_user(target_id) {
                bot.send_message(message.chat.id, "不能對項目維護人員或 Telegram 系統帳號執行此指令。").await?;
                return Ok(());
            }
            let reason = if reason.trim().is_empty() { "違反使用規則".to_string() } else { reason };
            runtime.set_user_banned(target_id, true, &reason, Some(from_id)).await.ok();
            log_maintainer_action(
                &bot,
                &runtime,
                from_id,
                &short_user(from),
                None,
                "/forbid",
                &format!("禁止 user_id={target_id} 使用本項目：{reason}"),
                UndoData::UserBanned { user_id: target_id },
            )
            .await;
            bot.send_message(message.chat.id, format!("已禁止 <code>{target_id}</code> 使用本項目的任何服務，並且無法再將本機器人加入群組。解除請用 <code>/forgive {target_id}</code>。")).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::Forgive(target) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以使用 /forgive。");
            let target = target.trim();
            let parsed = target.parse::<i64>().ok().or_else(|| real_reply(&message).and_then(|r| r.from.as_ref()).map(|u| u.id.0 as i64));
            let Some(id) = parsed else {
                bot.send_message(message.chat.id, "請提供 chat_id 或 user_id，例如 /forgive -1001234567890。").await?;
                return Ok(());
            };
            // Telegram group ids are negative and user ids positive, so one
            // command can lift either list without the caller having to say
            // which - matching how /leave and /forbid each take just an id.
            if id < 0 {
                if !runtime.is_group_banned(id).await {
                    bot.send_message(message.chat.id, format!("<code>{id}</code> 不在封禁群組名單中。")).parse_mode(ParseMode::Html).await?;
                    return Ok(());
                }
                runtime.set_group_banned(id, false, "", Some(from_id)).await.ok();
                log_maintainer_action(&bot, &runtime, from_id, &short_user(from), Some(id), "/forgive", &format!("解除群組封禁 {id}"), UndoData::NotRevertible).await;
                bot.send_message(message.chat.id, format!("已解除 <code>{id}</code> 的封禁，可以重新加入本機器人。")).parse_mode(ParseMode::Html).await?;
            } else {
                if !runtime.is_user_banned(id).await {
                    bot.send_message(message.chat.id, format!("<code>{id}</code> 不在封禁用戶名單中。")).parse_mode(ParseMode::Html).await?;
                    return Ok(());
                }
                runtime.set_user_banned(id, false, "", Some(from_id)).await.ok();
                log_maintainer_action(&bot, &runtime, from_id, &short_user(from), None, "/forgive", &format!("解除用戶封禁 {id}"), UndoData::NotRevertible).await;
                bot.send_message(message.chat.id, format!("已解除 <code>{id}</code> 的封禁。")).parse_mode(ParseMode::Html).await?;
            }
        }
        ModerationCommand::Whois(target) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以使用 /whois。");
            let target_id = target
                .trim()
                .parse::<i64>()
                .ok()
                .or_else(|| real_reply(&message).and_then(|r| r.from.as_ref()).map(|u| u.id.0 as i64));
            let Some(target_id) = target_id else {
                bot.send_message(message.chat.id, "請提供 user_id 或回覆該用戶，例如 /whois 12345。").await?;
                return Ok(());
            };

            let bans = runtime.find_active_bans_for_user(target_id).await.unwrap_or_default();
            let strikes = runtime.count_ban_strikes_for_user(target_id).await.unwrap_or(0);
            let netban = runtime.find_active_network_ban(target_id).await.ok().flatten();
            let global_wl = runtime.is_global_whitelisted(target_id).await.unwrap_or(false);
            let forbidden = runtime.is_user_banned(target_id).await;
            let reviewer = runtime.is_reviewer(target_id).await;
            let report_strikes = runtime.report_strikes(target_id).await;
            let maintainer = runtime.is_maintainer(target_id).await;

            let mut out = format!("<b>❖ 用戶查詢</b>\n<b>User ID</b>: <code>{target_id}</code>\n");
            let mut tags = Vec::new();
            if is_host(target_id) { tags.push("項目主持人"); }
            else if maintainer { tags.push("維護組"); }
            if reviewer { tags.push("審核員"); }
            if global_wl { tags.push("全域白名單"); }
            if forbidden { tags.push("已禁止使用本項目"); }
            if is_platform_pseudo_user(target_id) { tags.push("Telegram 系統帳號"); }
            out.push_str(&format!("<b>身分</b>: {}\n", if tags.is_empty() { "一般用戶".to_string() } else { tags.join(" / ") }));
            out.push_str(&format!("<b>歷史封禁次數</b>: {strikes}（含已撤銷）\n"));
            out.push_str(&format!(
                "<b>跨群組黑名單</b>: {}\n",
                match &netban {
                    Some(c) => format!("是（案例 <code>{}</code>）", c.id),
                    None => "否".to_string(),
                }
            ));
            out.push_str(&format!("<b>舉報被拒次數</b>: {report_strikes}/{REPORT_STRIKE_LIMIT}{}\n", if report_strikes >= REPORT_STRIKE_LIMIT { "（已暫停 /spam）" } else { "" }));

            if bans.is_empty() {
                out.push_str("\n<b>目前生效中的封禁</b>: 無");
            } else {
                out.push_str(&format!("\n<b>目前生效中的封禁</b>（{}）:", bans.len()));
                for case in bans.iter().take(10) {
                    out.push_str(&format!(
                        "\n• 群組 <code>{}</code> — {} <code>{}</code> <i>{}</i>",
                        case.chat_id,
                        chinese_case_action(case),
                        case.id,
                        utc8_display(case.created_at),
                    ));
                }
                if bans.len() > 10 {
                    out.push_str(&format!("\n…另有 {} 筆未顯示", bans.len() - 10));
                }
            }
            bot.send_message(message.chat.id, out).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::ReportReset(target) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以使用 /report_reset。");
            let target_id = target
                .trim()
                .parse::<i64>()
                .ok()
                .or_else(|| real_reply(&message).and_then(|r| r.from.as_ref()).map(|u| u.id.0 as i64));
            let Some(target_id) = target_id else {
                bot.send_message(message.chat.id, "請提供 user_id 或回覆該用戶。").await?;
                return Ok(());
            };
            let before = runtime.report_strikes(target_id).await;
            runtime.clear_report_strikes(target_id).await.ok();
            log_maintainer_action(&bot, &runtime, from_id, &short_user(from), None, "/report_reset", &format!("清除 user_id={target_id} 的舉報拒絕計數（原 {before}）"), UndoData::NotRevertible).await;
            bot.send_message(message.chat.id, format!("已清除 <code>{target_id}</code> 的舉報拒絕計數（原本 {before} 次），可以重新使用 /spam。")).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::MlEval(arg) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let holdout = arg.trim().parse::<f64>().ok().filter(|v| (0.05..=0.5).contains(v)).unwrap_or(0.2);
            match runtime.evaluate_model(holdout).await {
                Ok(report) => {
                    bot.send_message(message.chat.id, report).parse_mode(ParseMode::Html).await?;
                }
                Err(err) => {
                    bot.send_message(message.chat.id, format!("評估失敗：{err}")).await?;
                }
            }
        }
        ModerationCommand::ListBanned => {
            require_maintainer!(&bot, runtime, from_id, message, "只有維護人員可以使用此指令。");
            let (groups, users) = runtime.list_banned().await.unwrap_or_default();
            let mut out = String::from("<b>❖ 項目封禁名單</b>\n\n<b>群組</b>");
            if groups.is_empty() {
                out.push_str("\n（無）");
            }
            for (id, reason, created_at) in &groups {
                out.push_str(&format!("\n<code>{id}</code> — {} <i>{}</i>", escape_html(reason), escape_html(created_at)));
            }
            out.push_str("\n\n<b>用戶</b>");
            if users.is_empty() {
                out.push_str("\n（無）");
            }
            for (id, reason, created_at) in &users {
                out.push_str(&format!("\n<code>{id}</code> — {} <i>{}</i>", escape_html(reason), escape_html(created_at)));
            }
            bot.send_message(message.chat.id, out).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::Maintainer(sub, target) => {
            // Host only. Maintainers grant reviewers; only the host grants
            // maintainers, and the host itself can never be removed.
            if !is_host(from_id) {
                bot.send_message(message.chat.id, "只有項目主持人可以管理維護組。").await?;
                return Ok(());
            }
            let usage = "用法：/maintainer add|del <user_id>（或回覆該用戶）、/maintainer list";
            match sub.trim().to_lowercase().as_str() {
                "list" | "" => {
                    let ms = runtime.list_maintainers().await.unwrap_or_default();
                    let mut out = format!("<b>❖ 權限名單</b>\n\n<b>項目主持人</b>\n<code>{HOST_ID}</code>（不可移除）\n\n<b>維護組</b>");
                    if ms.is_empty() {
                        out.push_str("\n（無）");
                    }
                    for (id, created_at) in &ms {
                        out.push_str(&format!("\n<code>{id}</code> <i>{}</i>", escape_html(created_at)));
                    }
                    out.push_str("\n\n維護組可授予審核員、白名單等，但不能管理維護組本身。");
                    bot.send_message(message.chat.id, out).parse_mode(ParseMode::Html).await?;
                }
                verb @ ("add" | "del" | "remove") => {
                    let target_id = target
                        .trim()
                        .parse::<i64>()
                        .ok()
                        .or_else(|| real_reply(&message).and_then(|r| r.from.as_ref()).map(|u| u.id.0 as i64));
                    let Some(target_id) = target_id else {
                        bot.send_message(message.chat.id, usage).await?;
                        return Ok(());
                    };
                    if is_host(target_id) {
                        bot.send_message(message.chat.id, "項目主持人的權限無法透過指令變更。").await?;
                        return Ok(());
                    }
                    let enabled = verb == "add";
                    runtime.set_maintainer(target_id, enabled, Some(from_id)).await.ok();
                    log_maintainer_action(
                        &bot, &runtime, from_id, &short_user(from), None, "/maintainer",
                        &format!("{} 維護組 user_id={target_id}", if enabled { "新增" } else { "移除" }),
                        UndoData::Maintainer { user_id: target_id, old_enabled: !enabled },
                    )
                    .await;
                    bot.send_message(message.chat.id, format!("{} <code>{target_id}</code> 的維護組權限。", if enabled { "已授予" } else { "已撤銷" })).parse_mode(ParseMode::Html).await?;
                }
                _ => {
                    bot.send_message(message.chat.id, usage).await?;
                }
            }
        }
        ModerationCommand::Reviewer(sub, target) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以管理審核員。");
            let usage = "用法：/reviewer add|del <user_id>（或回覆該用戶）、/reviewer list";
            match sub.trim().to_lowercase().as_str() {
                "list" | "" => {
                    let reviewers = runtime.list_reviewers().await.unwrap_or_default();
                    let mut out = String::from("<b>❖ 審核員名單</b>\n");
                    if reviewers.is_empty() {
                        out.push_str("\n（無）");
                    }
                    for (id, created_at) in &reviewers {
                        out.push_str(&format!("\n<code>{id}</code> <i>{}</i>", escape_html(created_at)));
                    }
                    out.push_str("\n\n審核員可以處理舉報頻道的受理/拒絕與訓練批准，其餘維護指令不受影響。");
                    bot.send_message(message.chat.id, out).parse_mode(ParseMode::Html).await?;
                }
                verb @ ("add" | "del" | "remove") => {
                    let target_id = target
                        .trim()
                        .parse::<i64>()
                        .ok()
                        .or_else(|| real_reply(&message).and_then(|r| r.from.as_ref()).map(|u| u.id.0 as i64));
                    let Some(target_id) = target_id else {
                        bot.send_message(message.chat.id, usage).await?;
                        return Ok(());
                    };
                    let enabled = verb == "add";
                    runtime.set_reviewer(target_id, enabled, Some(from_id)).await.ok();
                    log_maintainer_action(
                        &bot,
                        &runtime,
                        from_id,
                        &short_user(from),
                        None,
                        "/reviewer",
                        &format!("{} 審核員 user_id={target_id}", if enabled { "新增" } else { "移除" }),
                        UndoData::Reviewer { user_id: target_id, old_enabled: !enabled },
                    )
                    .await;
                    let verb_text = if enabled { "已授予" } else { "已撤銷" };
                    bot.send_message(message.chat.id, format!("{verb_text} <code>{target_id}</code> 的審核員權限。")).parse_mode(ParseMode::Html).await?;
                }
                _ => {
                    bot.send_message(message.chat.id, usage).await?;
                }
            }
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

            if is_group_admin(&bot, message.chat.id, target_id).await || runtime.is_maintainer(target_id).await || is_platform_pseudo_user(target_id) {
                reply_ephemeral(&bot, &message, "不能對群組管理員或項目維護人員執行此指令。").await?;
                return Ok(());
            }

            let action = match cmd {
                ModerationCommand::SpamBan => ActionKind::SpamBan,
                ModerationCommand::Mute => ActionKind::Mute,
                ModerationCommand::Kick => ActionKind::Kick,
                _ => unreachable!(),
            };
            // A /sb on a message that is nothing but bot @-mentions is
            // guest-mode summon spam. Storing that as an ML sample would just
            // teach the model a username token; a regex rule catches the bot
            // handle deterministically instead. Detected here so the case
            // reason reflects it.
            let bot_spam = if matches!(action, ActionKind::SpamBan) {
                bot_mentions_only(&evidence_text)
            } else {
                None
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
                matched_rule_pattern: bot_spam.as_ref().map(|_| "BOTSPAM".to_string()),
                status: "done".to_string(),
                log_message_id: None,
                created_at: Utc::now(),
            };

            match action {
                ActionKind::SpamBan => {
                    let _ = bot.delete_message(message.chat.id, MessageId(source_id)).await;
                    ban_user(&bot, message.chat.id, target_id).await.ok();
                    // Deliberately does NOT train here. A group admin's /sb
                    // is a decision about their own room, and taking it as
                    // ground truth let anyone with admin rights anywhere
                    // write directly into the shared model. The text goes to
                    // the report channel instead and only trains once a
                    // maintainer approves it.
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
                broadcast_ban_status(&bot, &runtime, case.target_user_id, true).await;
                if bot_spam.is_some() {
                    // A bare username is not useful ML signal; the rule does
                    // the work, so skip the training-review queue.
                    capture_bot_spam_rules(&bot, &runtime, message.chat.id.0, &evidence_text).await;
                } else {
                    queue_training_review(&bot, &runtime, &case).await;
                }
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
        ModerationCommand::Pol(sub) => {
            if !message.chat.is_group() && !message.chat.is_supergroup() {
                return Ok(());
            }
            let chat_id = message.chat.id.0;
            let settings = runtime.get_group_modules(chat_id).await.unwrap_or_default();
            let authorized = settings.pol
                && (runtime.is_maintainer(from_id).await || is_group_admin(&bot, message.chat.id, from_id).await);
            if !authorized {
                // Unauthorized use - module not enabled here, or the caller
                // isn't an admin/maintainer - is completely silent. Deleting
                // the command and saying nothing is what keeps this module's
                // existence from ever being revealed to someone who isn't
                // already supposed to know about it.
                let _ = bot.delete_message(message.chat.id, message.id).await;
                return Ok(());
            }

            match sub.trim().to_lowercase().as_str() {
                "show" => {
                    let Some((target_id, target_name, _, _)) = extract_reply_context(&message).await else {
                        reply_ephemeral(&bot, &message, "請回覆一位用戶的訊息。").await?;
                        return Ok(());
                    };
                    let count = runtime.pol_warn_count(chat_id, target_id).await.unwrap_or(0);
                    bot.send_message(message.chat.id, format!("{} 目前在本群有 {count} 次警告。", escape_html(&target_name))).parse_mode(ParseMode::Html).await?;
                }
                "clear" => {
                    let Some((target_id, target_name, _, _)) = extract_reply_context(&message).await else {
                        reply_ephemeral(&bot, &message, "請回覆一位用戶的訊息。").await?;
                        return Ok(());
                    };
                    runtime.clear_pol_warns(chat_id, target_id).await.ok();
                    bot.send_message(message.chat.id, format!("已清除 {} 在本群的所有警告。", escape_html(&target_name))).parse_mode(ParseMode::Html).await?;
                }
                "" => {
                    let Some((target_id, target_name, source_id, _)) = extract_reply_context(&message).await else {
                        reply_ephemeral(&bot, &message, "請回覆一條訊息後再使用此指令。").await?;
                        return Ok(());
                    };
                    if is_group_admin(&bot, message.chat.id, target_id).await || runtime.is_maintainer(target_id).await || is_platform_pseudo_user(target_id) {
                        reply_ephemeral(&bot, &message, "不能對群組管理員或項目維護人員執行此指令。").await?;
                        return Ok(());
                    }

                    let _ = bot.delete_message(message.chat.id, MessageId(source_id)).await;
                    let prior_count = runtime.pol_warn_count(chat_id, target_id).await.unwrap_or(0);
                    let new_count = runtime.increment_pol_warn(chat_id, target_id).await.unwrap_or(prior_count + 1);

                    if prior_count == 0 {
                        // Warns never expire - the very next /pol on this user,
                        // whenever it happens, is a ban rather than a second warning.
                        let text = format!(
                            "{} 由于本群的公开性质及未来可能将与墙内社交平台连接，请勿讨论敏感内容，谢谢合作。为了群员的安全，您的原消息已撤回，警告一次，感谢理解。",
                            mention_link(target_id, &target_name),
                        );
                        let group_rules_button = InlineKeyboardMarkup::new(vec![vec![
                            InlineKeyboardButton::url("请按此查看群规", Url::parse("https://t.me/Chinese_wikimedia_activities/1/180").unwrap()),
                        ]]);
                        if let Ok(sent) = bot.send_message(message.chat.id, text).parse_mode(ParseMode::Html).reply_markup(group_rules_button).await {
                            let bot = bot.clone();
                            let warn_chat_id = message.chat.id;
                            let message_id = sent.id;
                            tokio::spawn(async move {
                                sleep(Duration::from_secs(24 * 60 * 60)).await;
                                let _ = bot.delete_message(warn_chat_id, message_id).await;
                            });
                        }
                    } else {
                        // Deliberately lightweight: a raw ban plus a private
                        // audit-log note, not a full CaseRecord - this stays
                        // outside /case, netban propagation, and the PM
                        // appeal bridge, matching this module's contained,
                        // non-public scope. /unban's existing case-less
                        // fallback already handles reversing it if needed.
                        let _ = ban_user(&bot, message.chat.id, target_id).await;
                        log_maintainer_action(
                            &bot, &runtime, from_id, &short_user(from), Some(chat_id), "/pol",
                            &format!("warn-pol 自動封禁 對象={target_id} warn_count={new_count}"),
                            UndoData::NotRevertible,
                        ).await;
                    }

                    let _ = bot.delete_message(message.chat.id, message.id).await;
                }
                _ => {
                    reply_ephemeral(&bot, &message, "用法：/pol（回覆訊息）、/pol show（回覆訊息）、/pol clear（回覆訊息）。").await?;
                }
            }
        }
        ModerationCommand::SpamReport => {
            // Three rejected reports and the command is gone. Maintainers are
            // exempt so a bad streak can't lock out the people who clear it.
            let strikes = runtime.report_strikes(from_id).await;
            if strikes >= REPORT_STRIKE_LIMIT && !runtime.is_maintainer(from_id).await {
                reply_ephemeral(
                    &bot,
                    &message,
                    format!("你已有 {strikes} 次舉報被拒絕，已暫停使用 /spam。如有疑問請透過 @SEELE_01_BOT 聯絡項目組。"),
                )
                .await?;
                let _ = bot.delete_message(message.chat.id, message.id).await;
                return Ok(());
            }

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

            bot
                .send_message(ChatId(runtime.config.report_channel_id), text)
                .parse_mode(ParseMode::Html)
                .reply_markup(keyboard)
                .await?;

            // log_message_id stays None here (not the report-review card's
            // message id, which lives in report_channel_id, a different
            // chat) - public_log_link() always builds its URL against
            // log_channel_id, so storing a different chat's message id
            // here previously produced a link to a numerically-coincidental
            // but completely unrelated log-channel message. It's only ever
            // meaningful once something actually calls log_action, which
            // only the "approve" review outcome does; "reject" correctly
            // leaves it unset, and /case already renders that as "-".
            store_case(&runtime, &case).await.ok();

            // Remember this confirmation so it can be updated to the outcome
            // once a reviewer decides. Reply to the report command so it's
            // clear whose report it belongs to.
            let sent = bot
                .send_message(message.chat.id, "已送交舉報處理頻道審核。")
                .reply_parameters(teloxide::types::ReplyParameters::new(message.id))
                .await?;
            let _ = runtime.set_report_confirmation(&case_id, message.chat.id.0, sent.id.0).await;
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
            let target_msg = real_reply(&message).unwrap_or(&message);
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
            let target_msg = real_reply(&message).unwrap_or(&message);
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
            let text = if let Some(target_msg) = real_reply(&message) {
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
        ModerationCommand::MlRetrain => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let (spam_docs, ham_docs) = runtime.retrain_from_samples().await.unwrap_or((0, 0));
            let rebuilt = runtime.rebuild_model().await.unwrap_or_default();
            let summary = format!(
                "已依目前的分詞規則重新計算全部詞頻：spam={spam_docs} ham={ham_docs}，詞彙量 {}",
                rebuilt.spam_tokens.len() + rebuilt.ham_tokens.len()
            );
            log_maintainer_action(&bot, &runtime, from_id, &short_user(from), None, "/ml_retrain", &summary, UndoData::NotRevertible).await;
            bot.send_message(message.chat.id, format!("{summary}\n\n所有詞頻皆重新由訓練樣本計算，不含任何人工偏置。")).await?;
        }
        ModerationCommand::MlDedupe => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");

            // Confirmed false positives from a manual audit of a training
            // export on 2026-07-31 (see conversation/commit history) -
            // ordinary messages that were mistakenly trained as spam.
            // Each is purged from spam and retrained as ham.
            const RECLASSIFY_TO_HAM: &[(&str, &str)] = &[
                ("aac024d0-3529-42d1-a71e-d74f99654620", "哪里人？"),
                ("6f1b5155-4d99-4297-a3af-d58884eac33b", "Apologize"),
                ("7bcf74ca-89e5-4423-b46f-b0dbc582d37c", "怎么排名"),
                ("9488f3ec-bf53-4026-883b-706780c4e0e1", "大家好，有没有人可以帮忙加百科的呀"),
            ];
            let mut reclassified = 0usize;
            for (case_id, text) in RECLASSIFY_TO_HAM {
                if runtime.purge_training_by_case(case_id).await.unwrap_or(0) > 0 && train_ham(&runtime, text, None).await.is_ok() {
                    reclassified += 1;
                }
            }

            let (dup_removed, empty_removed) = runtime.dedupe_training_samples().await.unwrap_or((0, 0));
            let rebuilt = runtime.rebuild_model().await.unwrap_or_default();

            let summary = format!(
                "重分類 {reclassified} 筆、移除重複樣本 {dup_removed} 筆、移除空白樣本 {empty_removed} 筆，重建後 spam_docs={} ham_docs={}",
                rebuilt.spam_docs, rebuilt.ham_docs,
            );
            log_maintainer_action(&bot, &runtime, from_id, &short_user(from), None, "/ml_dedupe", &summary, UndoData::NotRevertible).await;
            bot.send_message(message.chat.id, format!("已完成訓練資料清理。\n{summary}")).await?;
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
                        text.push_str("\n⚠️ 此 token 佔整體 spam 計數超過 20%，會拖累其他詞的判斷。通常是同一段文字被重複訓練所致，建議用 /ml_dedupe 去重，或用 /ml_purge_text 清除該樣本後再 /ml_retrain。");
                    }
                }
                if let Some((word, count)) = top_ham {
                    text.push_str(&format!("\n最大 ham token: <code>{}</code> = {count}", escape_html(&word)));
                    if ham > 0 && (count as f64) / (ham as f64) > 0.2 {
                        text.push_str("\n⚠️ 此 token 佔整體 ham 計數超過 20%，通常是同一段文字被重複訓練所致，建議用 /ml_dedupe 去重後再 /ml_retrain。");
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
        ModerationCommand::RefreshBL => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let Some(existing_id) = runtime.blacklist_reason_message_id().await.ok().flatten() else {
                bot.send_message(message.chat.id, "還沒發過封禁代號說明，先用 /updateBL。").await?;
                return Ok(());
            };
            let text = build_blacklist_reason_text(&runtime);
            match bot.edit_message_text(ChatId(runtime.config.log_channel_id), MessageId(existing_id), text).parse_mode(ParseMode::Html).await {
                Ok(_) => {
                    bot.send_message(message.chat.id, format!("已就地更新，沒有重新發文/釘選：<code>{}</code>", existing_id)).parse_mode(ParseMode::Html).await?;
                }
                Err(err) if err.to_string().contains("message is not modified") => {
                    // Telegram only returns this when the content is already
                    // byte-identical, i.e. the pinned message is up to date.
                    // That's success, not failure - don't push people to
                    // /updateBL and re-post a duplicate.
                    bot.send_message(message.chat.id, "封禁代號說明已是最新，無需變更。").await?;
                }
                Err(err) => {
                    bot.send_message(message.chat.id, format!("編輯失敗（訊息可能已被刪除），改用 /updateBL：{err}")).await?;
                }
            }
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

            // No module name (or an explicit "status"/"list"): report where
            // every public module currently stands instead of erroring out.
            if key.is_empty() || key == "status" || key == "list" {
                let mut lines = String::from("<b>本群模組狀態</b>\n");
                for (mod_key, display, baseline) in PUBLIC_MODULES {
                    let on = module_flag(&old_settings, mod_key).unwrap_or(false);
                    lines.push_str(&format!(
                        "\n{} <code>{display}</code>{}",
                        if on { "✅" } else { "❌" },
                        if *baseline { "（基礎防護）" } else { "" },
                    ));
                }
                lines.push_str("\n\n<code>/module &lt;名稱&gt; on|off</code> 可單獨切換，<code>/module all on|off</code> 可一次全開或全關。\n標示為基礎防護的模組不會被 <code>/module all off</code> 關閉，需要時請單獨關閉。");
                bot.send_message(message.chat.id, lines).parse_mode(ParseMode::Html).await?;
                return Ok(());
            }

            // Bulk toggle. Only ever touches PUBLIC_MODULES, so the
            // maintainer-gated warn-pol can't be switched on this way, and
            // a bulk *off* additionally skips the baseline protections -
            // see PUBLIC_MODULES.
            if key == "all" {
                if !matches!(state.to_lowercase().as_str(), "on" | "enable" | "enabled" | "off" | "disable" | "disabled") {
                    reply_ephemeral(&bot, &message, "請指定 on 或 off，例如 /module all on。").await?;
                    return Ok(());
                }
                let mut old = Vec::new();
                let mut skipped = Vec::new();
                for (mod_key, display, baseline) in PUBLIC_MODULES {
                    if !enabled && *baseline {
                        skipped.push(*display);
                        continue;
                    }
                    if let Some(was) = module_flag(&old_settings, mod_key) {
                        old.push((mod_key.to_string(), was));
                    }
                    runtime.set_group_module(message.chat.id.0, mod_key, enabled).await.ok();
                }
                log_maintainer_action(
                    &bot,
                    &runtime,
                    from_id,
                    &short_user(from),
                    Some(message.chat.id.0),
                    "/module",
                    &format!("all →{enabled}（{} 個模組）", old.len()),
                    UndoData::GroupModulesBulk { chat_id: message.chat.id.0, old: old.clone() },
                )
                .await;
                let mut reply = format!("已將 {} 個公開模組設為 {}。", old.len(), if enabled { "on" } else { "off" });
                if !skipped.is_empty() {
                    reply.push_str(&format!("\n基礎防護未關閉：{}（如確定要關，請單獨執行 /module <名稱> off）", skipped.join(" / ")));
                }
                bot.send_message(message.chat.id, reply).await?;
                return Ok(());
            }

            let old_enabled = module_flag(&old_settings, &key);
            // "warn-pol" is a customized module - not open for public use.
            // Enabling it (never disabling) requires the chat to already be
            // on the maintainer-only allowlist (set via /magic). If it
            // isn't, fall through to the exact same "unsupported module
            // name" error as any typo - anyone not already allowlisted
            // can't tell "warn-pol" is a real module from that message.
            if key == "warn-pol" && enabled && !runtime.is_module_allowed("warn-pol", message.chat.id.0).await.unwrap_or(false) {
                reply_ephemeral(&bot, &message, "模組名稱僅支援 NoHalal / NoSM / Flood / Captcha / Netban / CmdClean / GuestBan / NoContact / NoVoice / NoExec。").await?;
                return Ok(());
            }
            if old_enabled.is_none() {
                reply_ephemeral(&bot, &message, "模組名稱僅支援 NoHalal / NoSM / Flood / Captcha / Netban / CmdClean / GuestBan / NoContact / NoVoice / NoExec。").await?;
                return Ok(());
            }
            runtime.set_group_module(message.chat.id.0, &key, enabled).await.ok();
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
            let Some(user_id) = target.parse::<i64>().ok().or_else(|| real_reply(&message).and_then(|m| m.from.as_ref()).map(|u| u.id.0 as i64)) else {
                reply_ephemeral(&bot, &message, "請提供 userid 或回覆一位用戶。").await?;
                return Ok(());
            };
            let old_enabled = runtime.is_group_whitelisted(message.chat.id.0, user_id).await.unwrap_or(false);
            runtime.set_group_whitelist(message.chat.id.0, user_id, true, Some(from_id)).await.ok();
            log_maintainer_action(&bot, &runtime, from_id, &short_user(from), Some(message.chat.id.0), "/white", &format!("本群白名單 user_id={user_id} {old_enabled}→true"), UndoData::GroupWhitelist { chat_id: message.chat.id.0, user_id, old_enabled }).await;

            // Whitelisting someone who's currently banned in this group but
            // leaving them locked out defeats the point - a group admin
            // reaching for /white almost always means "let this person back
            // in and stop flagging them," not just the latter. Scoped to
            // this group's Telegram-level ban only, same as /unban's
            // non-maintainer path - no case/training-data changes here.
            let was_banned = bot
                .get_chat_member(message.chat.id, UserId(user_id as u64))
                .await
                .map(|m| m.kind.is_banned())
                .unwrap_or(false);
            if was_banned {
                let _ = bot.unban_chat_member(message.chat.id, UserId(user_id as u64)).await;
                broadcast_unban_if_fully_clear(&bot, &runtime, user_id).await;
                bot.send_message(message.chat.id, format!("已將 <code>{user_id}</code> 加入本群白名單，並解除其在本群的封禁。")).parse_mode(ParseMode::Html).await?;
            } else {
                bot.send_message(message.chat.id, format!("已將 <code>{user_id}</code> 加入本群白名單。",)).parse_mode(ParseMode::Html).await?;
            }

            // Delete the command message to minimize group disruption
            let _ = bot.delete_message(message.chat.id, message.id).await;
        }
        ModerationCommand::WhiteGlobal(target) => {
            require_maintainer!(&bot, runtime, from_id, message, "只有項目維護組可以使用此指令。");
            let Some(user_id) = target.parse::<i64>().ok().or_else(|| real_reply(&message).and_then(|m| m.from.as_ref()).map(|u| u.id.0 as i64)) else {
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
            let Some(user_id) = target.parse::<i64>().ok().or_else(|| real_reply(&message).and_then(|m| m.from.as_ref()).map(|u| u.id.0 as i64)) else {
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
            let Some(user_id) = target.parse::<i64>().ok().or_else(|| real_reply(&message).and_then(|m| m.from.as_ref()).map(|u| u.id.0 as i64)) else {
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
            let Some(target_msg) = real_reply(&message) else {
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
            let target_msg = real_reply(&message).unwrap_or(&message);
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
            if !message.chat.is_private() || !runtime.is_maintainer(from_id).await {
                bot.send_message(message.chat.id, "只允許維護者在私訊中啟動批量訓練。") .await?;
                return Ok(());
            }
            runtime.start_mass_train(from_id).await;
            runtime.set_mass_train_mode(from_id, &mode).await;
            bot.send_message(message.chat.id, format!("已啟動批量訓練，模式: {mode}。"))
                .await?;
        }
        ModerationCommand::MlStartMassHam => {
            if !message.chat.is_private() || !runtime.is_maintainer(from_id).await {
                bot.send_message(message.chat.id, "只允許維護者在私訊中啟動批量訓練。") .await?;
                return Ok(());
            }
            runtime.start_mass_train(from_id).await;
            runtime.set_mass_train_mode(from_id, "ham" ).await;
            bot.send_message(message.chat.id, "已啟動批量訓練，模式: ham。接下來你在這個私訊中傳送的純文本訊息會被收集；完成後使用 /ml_finish_mass_ham。")
                .await?;
        }
        ModerationCommand::MlDebugParse => {
            if !message.chat.is_private() || !runtime.is_maintainer(from_id).await {
                bot.send_message(message.chat.id, "只允許維護者在私訊中使用 /ml_debug_parse。") .await?;
                return Ok(());
            }
            let target_msg = real_reply(&message).unwrap_or(&message);
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
            let target_msg = real_reply(&message).unwrap_or(&message);
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
            if !message.chat.is_private() || !runtime.is_maintainer(from_id).await {
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
            if !message.chat.is_private() || !runtime.is_maintainer(from_id).await {
                bot.send_message(message.chat.id, "只允許維護者在私訊中結束批量訓練。") .await?;
                return Ok(());
            }
            let samples = runtime.finish_mass_train(from_id).await;
            let mut imported = Vec::new();
            let mut count = 0usize;
            // Each buffered entry is one *Telegram message*, which may
            // itself be a pasted multi-line block (e.g. a chat log dump) -
            // split on newlines so each original line becomes its own ham
            // sample, rather than training the whole pasted block as one
            // mixed-signal document.
            for sample in samples {
                for line in sample.lines() {
                    let line = line.trim();
                    if line.is_empty() { continue; }
                    imported.push(line.to_string());
                    train_ham(&runtime, line, None).await.ok();
                    count += 1;
                }
            }
            bot.send_message(message.chat.id, format!("批量訓練完成。ham: {count}\n\n已提取並訓練的字串：\n{}", if imported.is_empty() { "無可提取樣本".to_string() } else { imported.join("\n---\n") })).await?;
            runtime.clear_mass_train(from_id).await;
        }
        ModerationCommand::Unban(arg) => {
            let is_maintainer_user = runtime.is_maintainer(from_id).await;
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
                let target_user_id = if let Some(target_msg) = real_reply(&message) {
                    target_msg.from.as_ref().map(|u| u.id.0 as i64)
                } else {
                    arg.trim().parse::<i64>().ok()
                };
                let Some(target_user_id) = target_user_id else {
                    reply_ephemeral(&bot, &message, "請回覆要解封的用戶，或提供 user_id。").await?;
                    return Ok(());
                };
                if let Err(err) = bot.unban_chat_member(message.chat.id, UserId(target_user_id as u64)).await {
                    let Some(reason) = unban_noop_reason(&err) else {
                        bot.send_message(message.chat.id, format!("解封失敗：{err}")).await?;
                        return Ok(());
                    };
                    reply_ephemeral(&bot, &message, format!("該用戶目前並未被封禁（{reason}）。")).await?;
                    return Ok(());
                }
                broadcast_unban_if_fully_clear(&bot, &runtime, target_user_id).await;
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
            let resolved = if let Some(target_msg) = real_reply(&message) {
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

            // Reverse EVERY active ban this user has, wherever it happened -
            // not just one case in the current chat. Leaving another case
            // active kept the user on the blacklist, so check_reban_and_act
            // / check_netban_and_act would re-ban them in that group the
            // moment they posted again - the reported "unban then re-ban".
            let mut active = runtime.find_active_bans_for_user(target_user_id).await.unwrap_or_default();
            if let Some(c) = case_from_id {
                if !active.iter().any(|a| a.id == c.id) {
                    active.push(c);
                }
            }

            if active.is_empty() {
                // No case this project tracks - just lift a raw Telegram ban
                // in the resolved chat.
                if let Err(err) = bot.unban_chat_member(ChatId(chat_id), UserId(target_user_id as u64)).await {
                    let Some(reason) = unban_noop_reason(&err) else {
                        bot.send_message(message.chat.id, format!("解封失敗：{err}")).await?;
                        return Ok(());
                    };
                    bot.send_message(message.chat.id, format!("該用戶目前並未被封禁（{reason}），無需解封。")).await?;
                    return Ok(());
                }
                bot.send_message(
                    message.chat.id,
                    format!("已在本群解封用戶 <code>{target_user_id}</code>。（找不到本專案的封禁記錄，沒有訓練樣本需要清除）"),
                )
                .parse_mode(ParseMode::Html)
                .await?;
                return Ok(());
            }

            let total = active.len();
            let mut reversed = 0usize;
            let mut errors = Vec::new();
            for c in active {
                match reverse_ban_case(&bot, &runtime, c, from_id, &short_user(from)).await {
                    Ok(_) => reversed += 1,
                    Err(err) => errors.push(err),
                }
            }
            let mut reply = format!("已解封用戶 <code>{target_user_id}</code>，撤銷 {reversed}/{total} 筆封禁案例並清除對應訓練樣本，同時移出跨群組黑名單。");
            if !errors.is_empty() {
                reply.push_str(&format!("\n部分失敗：{}", errors.join("；")));
            }
            bot.send_message(message.chat.id, reply).parse_mode(ParseMode::Html).await?;
        }
        ModerationCommand::Unmute(arg) => {
            let is_maintainer_user = runtime.is_maintainer(from_id).await;
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
                let target_user_id = if let Some(target_msg) = real_reply(&message) {
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
            let resolved = if let Some(target_msg) = real_reply(&message) {
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
                UndoData::GroupModule { chat_id, module, old_enabled } => match runtime.set_group_module(chat_id, &module, old_enabled).await {
                    Ok(()) => Ok(format!("已將群組 {chat_id} 的模組 {module} 復原為 {old_enabled}。")),
                    Err(e) => Err(e.to_string()),
                },
                UndoData::GroupBanned { chat_id } => match runtime.set_group_banned(chat_id, false, "", None).await {
                    Ok(()) => Ok(format!("已解除群組 {chat_id} 的項目封禁（機器人仍需重新邀請才會回到該群）。")),
                    Err(e) => Err(e.to_string()),
                },
                UndoData::UserBanned { user_id } => match runtime.set_user_banned(user_id, false, "", None).await {
                    Ok(()) => Ok(format!("已解除 user_id={user_id} 的項目封禁。")),
                    Err(e) => Err(e.to_string()),
                },
                UndoData::Reviewer { user_id, old_enabled } => match runtime.set_reviewer(user_id, old_enabled, None).await {
                    Ok(()) => Ok(format!("已將 user_id={user_id} 的審核員權限復原為 {old_enabled}。")),
                    Err(e) => Err(e.to_string()),
                },
                UndoData::Maintainer { user_id, old_enabled } => match runtime.set_maintainer(user_id, old_enabled, None).await {
                    Ok(()) => Ok(format!("已將 user_id={user_id} 的維護組權限復原為 {old_enabled}。")),
                    Err(e) => Err(e.to_string()),
                },
                UndoData::GroupModulesBulk { chat_id, old } => {
                    let mut failed = Vec::new();
                    for (module, old_enabled) in &old {
                        if let Err(e) = runtime.set_group_module(chat_id, module, *old_enabled).await {
                            failed.push(format!("{module}: {e}"));
                        }
                    }
                    if failed.is_empty() {
                        Ok(format!("已將群組 {chat_id} 的 {} 個模組全部復原。", old.len()))
                    } else {
                        Err(failed.join("; "))
                    }
                }
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
    if !matches!(kind, "review" | "train") || case_id.is_empty() {
        return Ok(());
    }

    if q.message.as_ref().map(|m| m.chat().id.0 != runtime.config.report_channel_id).unwrap_or(true) {
        bot.answer_callback_query(q.id).text("此按鈕只能在舉報處理頻道使用").await?;
        return Ok(());
    }

    // Being able to see the report channel is not authority to act in it.
    // Every button here either bans someone across the network or writes
    // into the shared model, so it takes an explicit reviewer grant (or
    // maintainer). Checked once, before any decision branch, so no button
    // added later can miss it.
    if !runtime.is_maintainer(from_id).await && !runtime.is_reviewer(from_id).await {
        bot.answer_callback_query(q.id).text("只有審核員或維護組可以處理此項目").await?;
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

    // The /sb training queue. Unlike the "review" flow below, nothing about
    // the ban changes here either way - the ban already happened in the
    // group. This decides only whether the text is allowed into the model.
    if kind == "train" {
        let (note, toast) = match decision {
            "approve" => {
                if let Err(err) = train_spam(&runtime, &case.evidence_text, Some(&case.id)).await {
                    log_callback_error(&bot, &runtime, &case, "train_spam", &err.to_string()).await;
                    bot.answer_callback_query(q.id).text("訓練失敗").await?;
                    return Ok(());
                }
                // A reviewer approving a /sb is the second gate that promotes
                // it from a local ban to the shared blacklist.
                commit_network_ban(&bot, &runtime, &case).await;
                broadcast_ban_status(&bot, &runtime, case.target_user_id, true).await;
                ("已批准並寫入模型，且已加入專案黑名單", "已批准並加入黑名單")
            }
            "reject" => ("已拒絕，未寫入模型（封禁不受影響）", "已拒絕訓練"),
            _ => return Ok(()),
        };
        let body = format!(
            "<b>待審核訓練樣本</b>（來自 /sb）\n\n<b>對象</b>: {} (<code>{}</code>)\n<b>操作者</b>: {}\n<b>群組</b>: <code>{}</code>\n<b>內容</b>: <blockquote>{}</blockquote>\n<b>案例</b>: <code>{}</code>\n<b>狀態</b>: {note}\n<b>處理者</b>: <code>{from_id}</code>",
            escape_html(&case.target_name),
            case.target_user_id,
            escape_html(case.actor_name.as_deref().unwrap_or("unknown")),
            case.chat_id,
            escape_html(&case.evidence_text),
            case.id,
        );
        let _ = bot.edit_message_text(message.chat().id, message.id(), body).parse_mode(ParseMode::Html).await;
        let _ = bot.edit_message_reply_markup(message.chat().id, message.id()).await;
        bot.answer_callback_query(q.id).text(toast).await?;
        return Ok(());
    }

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
            broadcast_ban_status(&bot, &runtime, updated.target_user_id, true).await;
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
            if let Some((chat_id, msg_id)) = runtime.take_report_confirmation(&case.id).await {
                let _ = bot
                    .edit_message_text(ChatId(chat_id), MessageId(msg_id), "✅ 舉報已受理，對象已被封禁。")
                    .await;
            }
            bot.answer_callback_query(q.id).text("已受理並封禁").await?;
        }
        "reject" => {
            if let Err(err) = train_ham(&runtime, &case.evidence_text, Some(&case.id)).await {
                log_callback_error(&bot, &runtime, &case, "train_ham", &err.to_string()).await;
            }
            // Strike the reporter. Read from `case`, not `updated`: the
            // lines below overwrite actor_user_id with the reviewer.
            let mut strike_note = String::new();
            if let Some(reporter) = case.actor_user_id {
                if !runtime.is_maintainer(reporter).await {
                    let count = runtime.add_report_strike(reporter).await.unwrap_or(0);
                    strike_note = if count >= REPORT_STRIKE_LIMIT {
                        format!("\n<b>舉報者</b>: <code>{reporter}</code> 已累計 {count} 次被拒，已暫停使用 /spam")
                    } else {
                        format!("\n<b>舉報者</b>: <code>{reporter}</code> 已累計 {count}/{REPORT_STRIKE_LIMIT} 次被拒")
                    };
                }
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
                "<b>新的 /spam 申請</b>\n\n<b>對象</b>: {} ({})\n<b>發起人</b>: {}\n<b>內容</b>: <blockquote>{}</blockquote>\n<b>案例</b>: <code>{}</code>\n<b>狀態</b>: 已拒絕受理\n<b>處理者</b>: <code>{}</code>{}",
                escape_html(&case.target_name),
                case.target_user_id,
                escape_html(case.actor_name.as_deref().unwrap_or("unknown")),
                escape_html(&case.evidence_text),
                case.id,
                from_id,
                strike_note
            );
            let _ = bot.edit_message_text(message.chat().id, message.id(), body).parse_mode(ParseMode::Html).await;
            let _ = bot.edit_message_reply_markup(message.chat().id, message.id()).await;
            if let Some((chat_id, msg_id)) = runtime.take_report_confirmation(&case.id).await {
                let _ = bot
                    .edit_message_text(ChatId(chat_id), MessageId(msg_id), "此舉報未被受理。")
                    .await;
            }
            bot.answer_callback_query(q.id).text("已拒絕受理").await?;
        }
        _ => {}
    }

    Ok(())
}

async fn auto_moderate(bot: Bot, runtime: Arc<Runtime>, message: Message) -> ResponseResult<()> {
    let Some(user) = message.from.as_ref() else { return Ok(()); };
    if user.is_bot || is_platform_pseudo_user(user.id.0 as i64) {
        return Ok(());
    }
    if runtime.is_global_whitelisted(user.id.0 as i64).await.unwrap_or(false) {
        return Ok(());
    }
    if runtime.is_group_whitelisted(message.chat.id.0, user.id.0 as i64).await.unwrap_or(false) {
        return Ok(());
    }
    if is_group_admin(&bot, message.chat.id, user.id.0 as i64).await || runtime.is_maintainer(user.id.0 as i64).await {
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
            broadcast_ban_status(&bot, &runtime, updated.target_user_id, true).await;
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

    // A pure bot-mention message that scores as spam is bot-advert spam,
    // not just any ML hit - label it BOTSPAM and turn the handle into a
    // regex rule below, same as a manual /sb would.
    let is_bot_spam = bot_mentions_only(&text).is_some();

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
        matched_rule_pattern: Some(if is_bot_spam { "BOTSPAM" } else { "ML" }.to_string()),
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
    broadcast_ban_status(&bot, &runtime, case.target_user_id, true).await;
    if is_bot_spam {
        capture_bot_spam_rules(&bot, &runtime, message.chat.id.0, &text).await;
    }
    Ok(())
}

async fn score_only(bot: &Bot, runtime: &Runtime, message: &Message) -> ResponseResult<()> {
    let Some(user) = message.from.as_ref() else { return Ok(()); };
    if user.is_bot || is_platform_pseudo_user(user.id.0 as i64) {
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

async fn ensure_bot_can_moderate(bot: &Bot, runtime: &Runtime, chat_id: ChatId) -> ResponseResult<bool> {
    let Some(me_id) = runtime.me_id(bot).await else { return Ok(false) };
    let member = match bot.get_chat_member(chat_id, me_id).await {
        Ok(m) => m,
        Err(_) => {
            let _ = bot.send_message(chat_id, "機器人無法檢查權限，將退出此群。請確認管理員權限後再邀請。" ).await;
            let _ = bot.leave_chat(chat_id).await;
            return Ok(false);
        }
    };
    // Same reasoning as is_group_admin: ask the type, not the debug text.
    // This one also checks the two permissions the bot actually needs, so a
    // promotion that withheld them is caught here rather than surfacing
    // later as silently-failing deletes and bans.
    let allowed = member.kind.is_privileged()
        && member.kind.can_delete_messages()
        && member.kind.can_restrict_members();
    if !allowed {
        let _ = bot.send_message(chat_id, "機器人缺乏足夠的管理員權限，將退出此群。請確認至少具備刪訊息、封禁、解除封禁、禁言、踢出權限。" ).await;
        let _ = bot.leave_chat(chat_id).await;
    }
    Ok(allowed)
}

const EXCHANGE_SENDER_GBB: &str = "GBB";
const EXCHANGE_SENDER_PM: &str = "PM";

/// One "envelope" message on the shared SCP-079 exchange channel bus (see
/// https://scp-079.org/exchange/ for the generic convention). PM, a separate
/// ticket/appeal bot, posts these to ask "is this user banned, where, why"
/// and to request an unban after a maintainer accepts an appeal in its own
/// UI - GBB never talks to the appealing user directly, only answers PM.
#[derive(Debug, Deserialize)]
struct ExchangeEnvelope {
    from: String,
    #[serde(default)]
    to: Vec<String>,
    action: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    data: serde_json::Value,
}

/// PM's sender wraps its JSON in a Markdown code block; GBB's own sends
/// don't, but incoming text must tolerate either.
fn parse_exchange_envelope(text: &str) -> Option<ExchangeEnvelope> {
    let t = text.trim();
    let t = t.strip_prefix("```json").or_else(|| t.strip_prefix("```")).map(str::trim_start).unwrap_or(t);
    let t = t.strip_suffix("```").map(str::trim_end).unwrap_or(t);
    serde_json::from_str(t).ok()
}

/// No `parse_mode` - plain text, so JSON's braces/quotes are never
/// misinterpreted as Markdown/HTML.
async fn send_exchange_message(bot: &Bot, chat: i64, action: &str, kind: &str, data: serde_json::Value) {
    let text = serde_json::to_string_pretty(&serde_json::json!({
        "from": EXCHANGE_SENDER_GBB,
        "to": [EXCHANGE_SENDER_PM],
        "action": action,
        "type": kind,
        "data": data,
    }))
    .unwrap_or_default();
    let _ = bot.send_message(ChatId(chat), text).await;
}

async fn handle_exchange_query_bad(bot: &Bot, runtime: &Runtime, chat: i64, data: serde_json::Value) {
    let Some(user_id) = data.get("id").and_then(|v| v.as_i64()) else { return };
    let request_id = data.get("request_id").and_then(|v| v.as_str()).unwrap_or_default();
    let banned = runtime.find_active_bans_for_user(user_id).await.map(|cases| !cases.is_empty()).unwrap_or(false);
    // "report"/"bad" with an explicit is_banned field, mirroring bad_detail's
    // shape - not the old add/remove-verb encoding (easy to misread: which
    // verb means "banned" isn't obvious out of context).
    send_exchange_message(bot, chat, "report", "bad", serde_json::json!({ "id": user_id, "request_id": request_id, "is_banned": banned })).await;
}

async fn handle_exchange_query_bad_detail(bot: &Bot, runtime: &Runtime, chat: i64, data: serde_json::Value) {
    let Some(user_id) = data.get("id").and_then(|v| v.as_i64()) else { return };
    let request_id = data.get("request_id").and_then(|v| v.as_str()).unwrap_or_default().to_string();
    let cases = runtime.find_active_bans_for_user(user_id).await.unwrap_or_default();

    let Some(case) = cases.first() else {
        send_exchange_message(bot, chat, "report", "bad_detail", serde_json::json!({
            "id": user_id,
            "request_id": request_id,
            "is_banned": false,
        }))
        .await;
        return;
    };

    // GBB owns this taxonomy - PM never validates reason_code/ban_source/
    // risk_level against any enum, it just stores and displays them (only
    // reason_code is actually rendered).
    let reason_code = case.action.as_str().to_uppercase();
    let ban_source = match case.action.as_str() {
        "auto_ban" => "auto",
        "report_approved" => "report",
        _ => "manual",
    };
    let risk_level = match case.model_score {
        Some(score) if score >= 0.95 => "high",
        Some(score) if score >= 0.85 => "medium",
        Some(_) => "low",
        None => "high",
    };
    let banned_by = match case.actor_user_id {
        Some(id) => format!("admin:{id}"),
        None => "system".to_string(),
    };
    let excerpt: String = case.evidence_text.chars().take(300).collect();
    let chat_title = bot.get_chat(ChatId(case.chat_id)).await.ok().and_then(|c| c.title().map(str::to_string));
    // "How many times has this person been banned" (including bans later
    // lifted) - an unenforced field on PM's side, but the most natural
    // reading of "strike count" for a moderation history.
    let strike_count = runtime.count_ban_strikes_for_user(user_id).await.unwrap_or(0);

    send_exchange_message(bot, chat, "report", "bad_detail", serde_json::json!({
        "id": user_id,
        "request_id": request_id,
        "is_banned": true,
        "ban_source": ban_source,
        "reason_code": reason_code,
        "reason_text": case.evidence_text,
        "trigger_rule": case.matched_rule_pattern.clone().unwrap_or_default(),
        "trigger_content_excerpt": excerpt,
        "trigger_chat_id": case.chat_id,
        "trigger_chat_title": chat_title,
        "detected_at": case.created_at.to_rfc3339(),
        "banned_at": case.created_at.to_rfc3339(),
        "banned_by": banned_by,
        "risk_level": risk_level,
        "strike_count": strike_count,
        "evidence_refs": [format!("case_id:{}", case.id), format!("chat_id:{}", case.chat_id)],
        // PM's UI has zero enforcement on this field today - a no-op either way.
        "can_auto_unban": true,
    }))
    .await;
}

async fn handle_exchange_request_unban(bot: &Bot, runtime: &Runtime, chat: i64, data: serde_json::Value) {
    let Some(user_id) = data.get("id").and_then(|v| v.as_i64()) else { return };
    let request_id = data.get("request_id").and_then(|v| v.as_str()).unwrap_or_default().to_string();
    // Defensive defaults rather than rejecting outright - a legitimate
    // request shouldn't be dropped just because an optional-looking field
    // was missing.
    let operator_id = data.get("operator_id").and_then(|v| v.as_i64()).unwrap_or(0);
    let operator_name = data.get("operator_name").and_then(|v| v.as_str()).unwrap_or(EXCHANGE_SENDER_PM).to_string();

    let cases = runtime.find_active_bans_for_user(user_id).await.unwrap_or_default();
    if cases.is_empty() {
        // Idempotent: the desired end state ("not banned") already holds, and
        // PM has no retry path for a failed case, so erroring here would be
        // more likely to strand an appeal than to help.
        send_exchange_message(bot, chat, "result", "unban", serde_json::json!({
            "id": user_id,
            "request_id": request_id,
            "success": true,
            "message": "沒有找到有效的封禁記錄，視為已解封。",
        }))
        .await;
        return;
    }

    // A user can have independent active bans in several unrelated groups -
    // reverse all of them, attributing the reversal to the actual staff
    // member who accepted the appeal in PM, not to "PM" or "GBB".
    let mut reversed = Vec::new();
    let mut errors = Vec::new();
    for case in cases {
        let case_id = case.id.clone();
        match reverse_ban_case(bot, runtime, case, operator_id, &operator_name).await {
            Ok(_) => reversed.push(case_id),
            Err(err) => errors.push(format!("{case_id}: {err}")),
        }
    }

    let success = errors.is_empty();
    let summary = if success {
        format!("PM 申訴解封成功，共解除 {} 個 case：{}", reversed.len(), reversed.join(", "))
    } else {
        format!("PM 申訴解封部分失敗。成功：{}；失敗：{}", reversed.join(", "), errors.join("; "))
    };
    // A new, less-visible trust boundary (an external bot triggering state
    // change with no maintainer typing a command) - worth the audit-channel
    // visibility even though a human-typed /unban doesn't bother. Not
    // revertible: there's no "re-ban" mechanism.
    log_maintainer_action(bot, runtime, operator_id, &operator_name, None, "PM 申訴解封", &summary, UndoData::NotRevertible).await;

    send_exchange_message(bot, chat, "result", "unban", serde_json::json!({
        "id": user_id,
        "request_id": request_id,
        "success": success,
        "message": summary,
    }))
    .await;
}

async fn handle_exchange_post(bot: Bot, runtime: Arc<Runtime>, post: Message) -> ResponseResult<()> {
    let Some(exchange_chat) = runtime.exchange_channel().await else { return Ok(()) };
    if post.chat.id.0 != exchange_chat {
        return Ok(());
    }
    let Some(text) = post.text() else { return Ok(()) };
    let Some(envelope) = parse_exchange_envelope(text) else { return Ok(()) };
    // The same channel may carry traffic for/from PM's other siblings
    // (CLEAN, LONG, NOSPAM, ...) - anything not addressed to GBB from PM is
    // silently ignored, not an error.
    if envelope.from != EXCHANGE_SENDER_PM || !envelope.to.iter().any(|t| t == EXCHANGE_SENDER_GBB) {
        return Ok(());
    }

    match (envelope.action.as_str(), envelope.kind.as_str()) {
        ("query", "bad") => handle_exchange_query_bad(&bot, &runtime, exchange_chat, envelope.data).await,
        ("query", "bad_detail") => handle_exchange_query_bad_detail(&bot, &runtime, exchange_chat, envelope.data).await,
        ("request", "unban") => handle_exchange_request_unban(&bot, &runtime, exchange_chat, envelope.data).await,
        _ => {}
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // teloxide logs getUpdates failures, network retries and dispatcher
    // errors through the `log` crate. Without a backend those vanish - which
    // is why a stalled poller once went dark with no trace in kubectl logs.
    // Default to info so those diagnostics show; RUST_LOG can override.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("spb starting: {}", version_info_text().replace('\n', " "));

    let config = Config::from_env()?;
    let bot = Bot::new(config.bot_token.clone());
    let runtime = Arc::new(Runtime::load(config).await?);

    if let Some(owner_id) = runtime.config.owner_id {
        // Best-effort: a restart is exactly when this is most useful, but it
        // shouldn't block (or fail) startup if the DM can't be delivered -
        // e.g. the owner never having started a chat with the bot.
        let _ = bot.send_message(ChatId(owner_id), version_info_text()).parse_mode(ParseMode::Html).await;
    }

    // Liveness heartbeat. Every 30s, confirm the Telegram connection is
    // actually alive (get_me) and stamp a file next to the database. The
    // job's health check restarts the pod if this stops updating, so a
    // dead-but-Running poller no longer sits silently until someone notices.
    {
        let bot = bot.clone();
        let hb_path = runtime
            .config
            .sqlite_path
            .parent()
            .map(|p| p.join("spb.heartbeat"))
            .unwrap_or_else(|| PathBuf::from("spb.heartbeat"));
        tokio::spawn(async move {
            loop {
                match bot.get_me().await {
                    Ok(_) => {
                        if let Ok(dur) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                            let _ = tokio::fs::write(&hb_path, dur.as_secs().to_string()).await;
                        }
                    }
                    Err(err) => log::warn!("heartbeat get_me failed: {err}"),
                }
                sleep(Duration::from_secs(30)).await;
            }
        });
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
                // Service was terminated for this group (/leave). Nothing
                // else runs - no moderation, no commands - and the bot
                // removes itself again if it somehow got re-added, so the
                // termination sticks without a maintainer re-issuing it.
                if runtime.is_group_banned(message.chat.id.0).await {
                    let _ = bot.leave_chat(message.chat.id).await;
                    return Ok(());
                }

                unpin_channel_autopin(&bot, &runtime, &message).await;

                // Feeds check_guest_bot_and_act's invoker correlation - has
                // to run unconditionally, before any check below might
                // delete/short-circuit on this message, since a bare
                // `@thatbot` mention (the message we most need to catch)
                // never trips any other check on its own.
                if (message.chat.is_group() || message.chat.is_supergroup()) && !message.from.as_ref().map(|u| u.is_bot).unwrap_or(true) {
                    if let (Some(user), Some(text)) = (message.from.as_ref(), message.text().or(message.caption())) {
                        if !text.trim().is_empty() {
                            runtime.record_recent_message(message.chat.id.0, user.id.0 as i64, message.id, &short_user(user), text).await;
                        }
                    }
                }

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

                // Guest-mode spam: a message "from" a bot account that was
                // never actually added to this chat (see
                // check_guest_bot_and_act). Runs before netban/flood since
                // those already skip is_bot messages entirely and would
                // otherwise waste a lookup on every one.
                if runtime.config.test_group_id != Some(message.chat.id.0)
                    && check_guest_bot_and_act(&bot, &runtime, &message).await
                {
                    return Ok(());
                }

                // Netban safety net: catches members already in a group before
                // it opted in, or who joined between propagation events.
                if runtime.config.test_group_id != Some(message.chat.id.0)
                    && check_netban_and_act(&bot, &runtime, &message).await
                {
                    return Ok(());
                }

                // Same-chat ban-evasion safety net: this chat's own past ban
                // should already have stopped a repost, but if it didn't,
                // catch it here rather than falling through to ordinary
                // content scoring.
                if runtime.config.test_group_id != Some(message.chat.id.0)
                    && check_reban_and_act(&bot, &runtime, &message).await
                {
                    return Ok(());
                }

                // Attachment policy (contact card / voice note / executable).
                // Runs for every message type, before the content-based
                // checks below - those only look at text, which these
                // messages usually don't have.
                if runtime.config.test_group_id != Some(message.chat.id.0)
                    && check_attachment_policy_and_act(&bot, &runtime, &message).await
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

    let exchange_handler = Update::filter_channel_post().endpoint({
        let runtime = runtime.clone();
        move |bot: Bot, post: Message| {
            let runtime = runtime.clone();
            async move { handle_exchange_post(bot, runtime, post).await }
        }
    });

    let handler = dptree::entry()
        .branch(message_handler)
        .branch(callback_handler)
        .branch(exchange_handler);

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
            data_dir: dir.clone(),
            sqlite_path: dir.join("bot.db"),
            spam_threshold: 0.85,
            owner_id: None,
            hostctl_secret: None,
            hostctl_cmd: "/hostctl".to_string(),
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

    // Regression check for the spam_docs/ham_docs=0 incident found live in
    // production after /ml_dedupe: a prior bulk data-cleanup pass had
    // inserted rows straight into training_samples (bypassing train_spam/
    // train_ham's increment), so model_meta's spam_docs/ham_docs counters
    // never reflected the real row count - dedup's decrement-by-1-per-
    // removed-row then clamped them to 0 well before accounting for every
    // real row. Simulates exactly that: seed training_samples directly with
    // real rows while model_meta's counters stay wrong/stale, and confirm
    // rebuild_model (and by extension load_model, the startup path) derives
    // spam_docs/ham_docs from the actual training_samples rows rather than
    // trusting the drifted counter.
    #[tokio::test]
    async fn rebuild_model_derives_doc_counts_from_training_samples_not_stale_model_meta() {
        let runtime = test_runtime().await;
        runtime
            .with_conn(|conn| {
                conn.execute("INSERT INTO training_samples (label, text, created_at) VALUES ('spam', 'a', '2026-01-01')", [])?;
                conn.execute("INSERT INTO training_samples (label, text, created_at) VALUES ('spam', 'b', '2026-01-01')", [])?;
                conn.execute("INSERT INTO training_samples (label, text, created_at) VALUES ('ham', 'c', '2026-01-01')", [])?;
                // Stale/wrong on purpose - this is what a bulk import that
                // skipped train_spam/train_ham would leave behind.
                conn.execute("INSERT INTO model_meta (key, value) VALUES ('spam_docs', '0')", [])?;
                conn.execute("INSERT INTO model_meta (key, value) VALUES ('ham_docs', '0')", [])?;
                Ok(())
            })
            .await
            .unwrap();

        let rebuilt = runtime.rebuild_model().await.unwrap();
        assert_eq!(rebuilt.spam_docs, 2);
        assert_eq!(rebuilt.ham_docs, 1);
    }

    // Regression check for the token-count corruption incident (caused by
    // the since-removed /set command): seeds a v1-shaped DB
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
            data_dir: dir.clone(),
            sqlite_path: db_path.clone(),
            spam_threshold: 0.85,
            owner_id: None,
            hostctl_secret: None,
            hostctl_cmd: "/hostctl".to_string(),
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

    // Netban membership is now a property the case carries, set once at ban
    // time, not something re-derived from the origin group's live settings.
    // A group toggling its own netban switch must not retroactively promote
    // or retract bans it already made.
    #[tokio::test]
    async fn find_active_network_ban_uses_the_stored_flag_not_live_group_settings() {
        let runtime = test_runtime().await;
        let case = dummy_case(ActionKind::AutoBan, 100, 200, Utc::now());
        runtime.persist_case(&case).await.unwrap();

        assert!(
            runtime.find_active_network_ban(200).await.unwrap().is_none(),
            "a case that never made it onto the blacklist shouldn't count as a network ban"
        );

        // Flipping the origin group's switch after the fact changes nothing.
        runtime.set_group_module(100, "netban", true).await.unwrap();
        assert!(
            runtime.find_active_network_ban(200).await.unwrap().is_none(),
            "turning netban on later must not retroactively promote past bans"
        );

        runtime.mark_netban_eligible(&case.id).await.unwrap();
        assert_eq!(runtime.find_active_network_ban(200).await.unwrap().map(|c| c.id), Some(case.id.clone()));

        // ...and turning it back off must not retract what's already listed.
        runtime.set_group_module(100, "netban", false).await.unwrap();
        assert_eq!(runtime.find_active_network_ban(200).await.unwrap().map(|c| c.id), Some(case.id.clone()));
    }

    // Exercises the real v8→v9 upgrade against existing case history, since
    // getting the backfill wrong on the live DB either drops everyone off
    // the blacklist or promotes bans that never belonged on it. Builds a v9
    // DB, seeds cases, then rewinds it to v8 (drop the column, reset
    // user_version) and reloads so the migration actually runs.
    #[tokio::test]
    async fn migrate_v8_to_v9_backfills_existing_netbans() {
        let dir = std::env::temp_dir().join(format!("spb_test_{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let db_path = dir.join("bot.db");
        let config = Config {
            bot_token: "test".to_string(),
            log_channel_id: -1,
            report_channel_id: -1,
            test_group_id: None,
            data_dir: dir.clone(),
            sqlite_path: db_path.clone(),
            spam_threshold: 0.85,
            owner_id: None,
            hostctl_secret: None,
            hostctl_cmd: "/hostctl".to_string(),
        };

        // A maintainer-approved report: project-level, stays listed.
        let approved = dummy_case(ActionKind::ReportApproved, 100, 201, Utc::now());
        // Clears the global bar on score alone.
        let mut high_score = dummy_case(ActionKind::AutoBan, 300, 202, Utc::now());
        high_score.model_score = Some(0.97);
        // Below the bar - exactly what a lowered group threshold would have
        // produced. Must stay off the blacklist.
        let mut low_score = dummy_case(ActionKind::AutoBan, 300, 203, Utc::now());
        low_score.model_score = Some(0.20);
        // A group admin's /sb from a netban group. The old JOIN listed this;
        // the new rule must drop it, since that's the whole point.
        let manual_sb = dummy_case(ActionKind::SpamBan, 100, 204, Utc::now());
        // An unscored module ban (NoHalal / NoLongName) - group policy.
        let module_ban = dummy_case(ActionKind::AutoBan, 100, 205, Utc::now());

        {
            let runtime = Runtime::load(config.clone()).await.unwrap();
            runtime.set_group_module(100, "netban", true).await.unwrap();
            runtime.set_group_module(300, "netban", false).await.unwrap();
            for case in [&approved, &high_score, &low_score, &manual_sb, &module_ban] {
                runtime.persist_case(case).await.unwrap();
            }
            // Rewind to the pre-migration shape.
            runtime
                .with_conn(|conn| {
                    conn.execute("ALTER TABLE cases DROP COLUMN netban_eligible", [])?;
                    conn.execute("PRAGMA user_version = 8", [])?;
                    Ok(())
                })
                .await
                .unwrap();
        }

        let runtime = Runtime::load(config).await.unwrap();
        assert_eq!(
            runtime.find_active_network_ban(201).await.unwrap().map(|c| c.id),
            Some(approved.id),
            "a maintainer-approved report must stay listed"
        );
        assert_eq!(
            runtime.find_active_network_ban(202).await.unwrap().map(|c| c.id),
            Some(high_score.id),
            "a past ban clearing the global bar on score should be admitted"
        );
        assert!(
            runtime.find_active_network_ban(203).await.unwrap().is_none(),
            "a below-bar ban must not be backfilled onto the blacklist"
        );
        assert!(
            runtime.find_active_network_ban(204).await.unwrap().is_none(),
            "a group admin's /sb must be dropped from the blacklist, netban group or not"
        );
        assert!(
            runtime.find_active_network_ban(205).await.unwrap().is_none(),
            "an unscored module ban is group policy and must be dropped"
        );
    }

    // Backs /leave, /forbid and /forgive. The in-memory sets are what every
    // hot-path check actually reads, so a write that lands in SQLite but not
    // the cache (or vice versa) would leave a banned group being served
    // until the next restart - both must move together, in both directions.
    #[tokio::test]
    async fn project_denial_lists_round_trip_through_cache_and_db() {
        let runtime = test_runtime().await;
        assert!(!runtime.is_group_banned(-100).await);
        assert!(!runtime.is_user_banned(555).await);

        runtime.set_group_banned(-100, true, "abuse", Some(1)).await.unwrap();
        runtime.set_user_banned(555, true, "abuse", Some(1)).await.unwrap();
        assert!(runtime.is_group_banned(-100).await);
        assert!(runtime.is_user_banned(555).await);
        // Scoped: banning one must not implicate anyone else.
        assert!(!runtime.is_group_banned(-200).await);
        assert!(!runtime.is_user_banned(666).await);

        let (groups, users) = runtime.list_banned().await.unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, -100);
        assert_eq!(groups[0].1, "abuse");
        assert_eq!(users.len(), 1);
        assert_eq!(users[0].0, 555);

        runtime.set_group_banned(-100, false, "", Some(1)).await.unwrap();
        runtime.set_user_banned(555, false, "", Some(1)).await.unwrap();
        assert!(!runtime.is_group_banned(-100).await);
        assert!(!runtime.is_user_banned(555).await);
        let (groups, users) = runtime.list_banned().await.unwrap();
        assert!(groups.is_empty() && users.is_empty());
    }


    // GroupAnonymousBot and the channel-post sender are flagged is_bot and
    // are genuinely absent from the member list, which is exactly the
    // signature check_guest_bot_and_act uses - so without an explicit guard
    // it bans whichever real admin was posting anonymously. Global
    // whitelisting them works but shouldn't be required.
    #[test]
    fn anonymous_admin_pseudo_accounts_are_not_guest_bots() {
        assert!(is_platform_pseudo_user(1087968824), "GroupAnonymousBot - anonymous admin posts");
        assert!(is_platform_pseudo_user(136817688), "Channel - posts sent as a channel");
        // A real guest-mode spam bot has an ordinary id and must still be caught.
        assert!(!is_platform_pseudo_user(8631161519));
    }

    // The termination notice used to render "違反本項目的使用規範：違反使用規則"
    // - the placeholder reason restating the sentence it was attached to.
    // The rules reference is now a link to the published terms, and a
    // specific reason only appears when one was actually given.
    #[test]
    fn termination_notice_links_the_terms_and_drops_the_placeholder_reason() {
        let bare = service_termination_text(&classify_termination_reason(""));
        assert!(bare.contains(&format!("<a href=\"{TERMS_URL}\">使用規範</a>")), "the rules reference must link to the published terms");
        assert!(!bare.contains("具體事由"), "no reason given - the notice must not invent a detail line");

        let detailed = service_termination_text(&classify_termination_reason("大量濫用檢舉指令"));
        assert!(detailed.contains("<b>具體事由</b>：大量濫用檢舉指令"));

        // Whitespace-only is the same as nothing.
        assert!(!service_termination_text(&classify_termination_reason("   ")).contains("具體事由"));
        // A reason is user-supplied text going into an HTML message.
        assert!(service_termination_text(&classify_termination_reason("<b>x</b>")).contains("&lt;b&gt;x&lt;/b&gt;"));
    }

    // Reason codes must resolve loosely (case/punctuation-insensitive) and
    // link the detail line straight to the matching ToU section, while
    // unknown input stays free text pointing at no specific clause.
    #[test]
    fn leave_reason_codes_map_to_terms_sections() {
        for code in ["TnS", "T&S", "trust and safety", "TnS/T&S"] {
            let r = classify_termination_reason(code);
            assert_eq!(r.anchor, Some("termination"), "{code} should map to the termination section");
            let text = service_termination_text(&r);
            assert!(text.contains(&format!("href=\"{TERMS_URL}#termination\"")), "{code} must link the detail line to #termination");
            assert!(text.contains("信任與安全"), "{code} must render the Trust & Safety label");
        }
        assert_eq!(classify_termination_reason("private").anchor, Some("eligibility"));
        assert_eq!(classify_termination_reason("PG").anchor, Some("eligibility"));

        // Unknown / free text keeps the words and links to no specific clause.
        let free = classify_termination_reason("大量濫用檢舉指令");
        assert_eq!(free.anchor, None);
        assert_eq!(free.label, "大量濫用檢舉指令");
    }

    // The invite prompt and the terms link both build a URL button, which
    // panics on a malformed URL - a typo would take down every invite.
    #[test]
    fn terms_url_is_parseable_and_is_not_the_repository() {
        assert!(Url::parse(TERMS_URL).is_ok());
        assert!(!TERMS_URL.contains("github.com"), "link to the published page, not the source repository");
    }

    // Training one forwarded advert taught the model that "external",
    // "origin", "chat", "id", "username", "reply" and "channel" are spam,
    // because extract_full_text's `[external_reply_origin_channel_id: …]`
    // markers were tokenized as prose (`_` is punctuation, so the key
    // shattered into words). Those words then appeared on every forward, and
    // several are ordinary English.
    #[test]
    fn origin_markers_never_tokenize_into_prose_words() {
        let meta = "[external_origin_chat_id: -1001825797691]\n[external_origin_username: NGHYGFPD]\n[external_reply_origin_channel_id: -1001825797691]\n[fwd_id: -1001825797691]\n[fwd_user: NGHYGFPD]";
        let tokens = tokenize(meta);
        for leaked in ["external", "origin", "chat", "id", "username", "reply", "channel", "fwd"] {
            assert!(!tokens.contains(&leaked.to_string()), "marker key leaked as the prose token {leaked:?}: {tokens:?}");
        }
        // The identity is real signal and must survive, as one opaque token.
        assert!(tokens.contains(&"tgsrc1001825797691".to_string()), "channel id should survive: {tokens:?}");
        assert!(tokens.contains(&"tgsrcnghygfpd".to_string()), "channel username should survive: {tokens:?}");
    }

    // The other half of the misfire: ordinary English must not share tokens
    // with the marker keys, or a member asking a normal question inherits
    // whatever weight the adverts trained.
    #[test]
    fn ordinary_prose_does_not_collide_with_origin_markers() {
        let prose = tokenize("please reply in the chat, whats your username");
        let marker = tokenize("[external_origin_username: NGHYGFPD]");
        for token in &prose {
            assert!(!marker.contains(token), "prose token {token:?} also produced by a marker: {marker:?}");
        }
    }

    // The real message body still has to be scored - collapsing the markers
    // must not swallow the advert text they were appended to.
    #[test]
    fn collapsing_markers_keeps_the_message_body() {
        let tokens = tokenize("免费代理频道 大水必红\n[fwd_user: NGHYGFPD]");
        assert!(tokens.iter().any(|t| t.contains('免') || t.contains('代')), "body tokens missing: {tokens:?}");
        assert!(tokens.contains(&"tgsrcnghygfpd".to_string()));
    }

    // /ml_retrain exists because changing the tokenizer can't fix counts that
    // are already stored - rebuild_model only re-reads them. Replaying the
    // samples has to actually clear the old tokens out.
    #[tokio::test]
    async fn retrain_from_samples_reruns_the_current_tokenizer() {
        let runtime = test_runtime().await;
        train_spam(&runtime, "免费代理 大水必红", Some("case-1")).await.unwrap();
        train_ham(&runtime, "大家好 請問有人可以幫忙嗎", Some("case-2")).await.unwrap();

        // Simulate a count left behind by an older tokenizer: a word with no
        // backing sample at all.
        runtime
            .with_conn(|conn| {
                conn.execute("INSERT INTO word_frequencies (word, spam_count, ham_count) VALUES ('external', 99, 0)", [])?;
                Ok(())
            })
            .await
            .unwrap();

        let (spam_docs, ham_docs) = runtime.retrain_from_samples().await.unwrap();
        assert_eq!((spam_docs, ham_docs), (1, 1));

        let rebuilt = runtime.rebuild_model().await.unwrap();
        assert!(!rebuilt.spam_tokens.contains_key("external"), "a token with no backing sample must not survive a retrain");
        assert_eq!(rebuilt.spam_docs, 1);
        assert_eq!(rebuilt.ham_docs, 1);
        // Real samples are still represented.
        assert!(!rebuilt.spam_tokens.is_empty() && !rebuilt.ham_tokens.is_empty());
    }

    // A member could become a group admin by renaming themselves. The old
    // check formatted the whole ChatMember - profile included - and searched
    // that text for "Administrator"/"Owner", so a first name of "Owner"
    // passed while `kind` was `Member`. That handed the full admin command
    // set to anyone who edited their display name.
    #[test]
    fn membership_is_read_from_the_type_not_the_display_name() {
        let spoof: teloxide::types::ChatMember =
            serde_json::from_str(r#"{"user":{"id":123,"is_bot":false,"first_name":"Owner"},"status":"member"}"#).unwrap();
        assert!(!spoof.kind.is_privileged(), "a member named Owner is still just a member");
        assert!(
            format!("{spoof:?}").contains("Owner"),
            "the debug text really does contain the word - which is exactly why matching on it was unsafe"
        );

        let admin_spoof: teloxide::types::ChatMember =
            serde_json::from_str(r#"{"user":{"id":123,"is_bot":false,"first_name":"Administrator"},"status":"member"}"#).unwrap();
        assert!(!admin_spoof.kind.is_privileged());

        // A real administrator must still pass.
        let real: teloxide::types::ChatMember = serde_json::from_str(
            r#"{"user":{"id":9,"is_bot":false,"first_name":"Real"},"status":"administrator","can_be_edited":false,"is_anonymous":false,"can_manage_chat":true,"can_delete_messages":true,"can_manage_video_chats":true,"can_restrict_members":true,"can_promote_members":false,"can_change_info":true,"can_invite_users":true,"can_post_stories":false,"can_edit_stories":false,"can_delete_stories":false}"#,
        )
        .unwrap();
        assert!(real.kind.is_privileged());
        assert!(real.kind.can_delete_messages() && real.kind.can_restrict_members());

        // An owner always passes, and always has the permissions.
        let owner: teloxide::types::ChatMember =
            serde_json::from_str(r#"{"user":{"id":8,"is_bot":false,"first_name":"O"},"status":"creator","is_anonymous":false}"#).unwrap();
        assert!(owner.kind.is_privileged() && owner.kind.can_delete_messages());
    }

    // The three attachment modules ban on message type, so the
    // classification has to be exact: each fires only for its own kind, only
    // when that module is on, and never for an ordinary message.
    #[test]
    fn attachment_violation_matches_only_the_enabled_module() {
        let contact = r#"{"message_id":1,"date":1,"chat":{"id":-100,"type":"supergroup","title":"g"},"from":{"id":5,"is_bot":false,"first_name":"a"},"contact":{"phone_number":"+1","first_name":"x"}}"#;
        let voice = r#"{"message_id":2,"date":1,"chat":{"id":-100,"type":"supergroup","title":"g"},"from":{"id":5,"is_bot":false,"first_name":"a"},"voice":{"file_id":"f","file_unique_id":"u","duration":3,"mime_type":null}}"#;
        let exe = r#"{"message_id":3,"date":1,"chat":{"id":-100,"type":"supergroup","title":"g"},"from":{"id":5,"is_bot":false,"first_name":"a"},"document":{"file_id":"f","file_unique_id":"u","file_name":"invoice.exe"}}"#;
        let pdf = r#"{"message_id":4,"date":1,"chat":{"id":-100,"type":"supergroup","title":"g"},"from":{"id":5,"is_bot":false,"first_name":"a"},"document":{"file_id":"f","file_unique_id":"u","file_name":"notes.pdf"}}"#;
        let text = r#"{"message_id":5,"date":1,"chat":{"id":-100,"type":"supergroup","title":"g"},"from":{"id":5,"is_bot":false,"first_name":"a"},"text":"hello"}"#;
        let m = |j: &str| serde_json::from_str::<Message>(j).unwrap();

        let off = GroupModuleSettings::default();
        for j in [contact, voice, exe, pdf, text] {
            assert_eq!(attachment_violation(&m(j), &off), None, "modules default off - nothing should trip");
        }

        let all_on = GroupModuleSettings { no_contact: true, no_voice: true, no_exec: true, ..Default::default() };
        assert_eq!(attachment_violation(&m(contact), &all_on), Some("CONTACT"));
        assert_eq!(attachment_violation(&m(voice), &all_on), Some("VOICE"));
        assert_eq!(attachment_violation(&m(exe), &all_on), Some("EXEC_FILE"));
        assert_eq!(attachment_violation(&m(pdf), &all_on), None, "an ordinary document must not trip NoExec");
        assert_eq!(attachment_violation(&m(text), &all_on), None);

        // Each module is independent.
        let only_contact = GroupModuleSettings { no_contact: true, ..Default::default() };
        assert_eq!(attachment_violation(&m(contact), &only_contact), Some("CONTACT"));
        assert_eq!(attachment_violation(&m(voice), &only_contact), None);
        assert_eq!(attachment_violation(&m(exe), &only_contact), None);
    }

    // Extension parsing decides whether a file is treated as executable, so
    // the edge cases matter: trailing dots, no extension, and the fact that
    // the check is case-insensitive.
    #[test]
    fn file_extension_handles_awkward_names() {
        assert_eq!(file_extension("a.EXE").as_deref(), Some("exe"));
        assert_eq!(file_extension("archive.tar.gz").as_deref(), Some("gz"));
        assert_eq!(file_extension("trailing.exe.").as_deref(), Some("exe"));
        assert_eq!(file_extension("noext"), None);
        assert_eq!(file_extension("weird.e xe"), None);
        assert!(EXECUTABLE_EXTENSIONS.contains(&"apk") && EXECUTABLE_EXTENSIONS.contains(&"msi"));
        assert!(!EXECUTABLE_EXTENSIONS.contains(&"pdf") && !EXECUTABLE_EXTENSIONS.contains(&"zip"));
    }

    // /spam's three-strike rule: strikes accumulate per reporter, and
    // /report_reset clears them.
    #[tokio::test]
    async fn report_strikes_accumulate_and_reset() {
        let runtime = test_runtime().await;
        assert_eq!(runtime.report_strikes(555).await, 0);

        assert_eq!(runtime.add_report_strike(555).await.unwrap(), 1);
        assert_eq!(runtime.add_report_strike(555).await.unwrap(), 2);
        let third = runtime.add_report_strike(555).await.unwrap();
        assert_eq!(third, REPORT_STRIKE_LIMIT, "third rejection should hit the limit");
        assert_eq!(runtime.report_strikes(666).await, 0, "strikes are per reporter");

        runtime.clear_report_strikes(555).await.unwrap();
        assert_eq!(runtime.report_strikes(555).await, 0);
    }

    // /ml_eval must never touch the live model, and must actually separate
    // train from test - scoring a model on its own training rows would just
    // measure memorisation.
    #[tokio::test]
    async fn ml_eval_is_read_only_and_reports_thresholds() {
        let runtime = test_runtime().await;
        for i in 0..15 {
            train_spam(&runtime, &format!("免费代理 大水必红 赢钱 {i}"), None).await.unwrap();
            train_ham(&runtime, &format!("大家好 請問有人可以幫忙編輯條目嗎 {i}"), None).await.unwrap();
        }
        let before = runtime.rebuild_model().await.unwrap();

        let report = runtime.evaluate_model(0.2).await.unwrap();
        assert!(report.contains("模型評估"), "got: {report}");
        assert!(report.contains("0.85"), "the live threshold row should appear: {report}");

        let after = runtime.rebuild_model().await.unwrap();
        assert_eq!(before.spam_docs, after.spam_docs, "evaluation must not change the model");
        assert_eq!(before.spam_tokens.len(), after.spam_tokens.len());
    }

    // An accepted appeal reported back to PM as "failed" because Telegram
    // refused the unban with "user is an administrator of the chat" - which
    // means the user is not banned, i.e. the thing being asked for already
    // holds. reverse_ban_case then bailed before closing the case, so the
    // user stayed on the netban list with the bad training sample intact.
    #[test]
    fn unban_errors_that_mean_already_not_banned_are_not_failures() {
        let noop = |msg: &str| {
            let err = teloxide::RequestError::Api(teloxide::ApiError::Unknown(msg.to_string()));
            unban_noop_reason(&err)
        };

        // The exact string from the incident.
        assert!(noop("Bad Request: user is an administrator of the chat").is_some());
        assert!(noop("Bad Request: chat not found").is_some());
        assert!(noop("Forbidden: bot was kicked from the supergroup chat").is_some());
        assert!(noop("Bad Request: user not found").is_some());
        assert!(noop("Bad Request: PARTICIPANT_ID_INVALID").is_some());

        // Genuine failures must still be reported, not silently swallowed -
        // here the ban really is still in place.
        assert!(noop("Bad Request: not enough rights to restrict/unrestrict chat member").is_none());
        assert!(noop("Too Many Requests: retry after 30").is_none());
        assert!(noop("Bad Request: something entirely new").is_none());
    }

    // Telegram rejects a sendMessage over 4096 characters, and these grow
    // every time a command is added - /help_op was at 3,000 before being
    // split into sections. A help text that has quietly outgrown the limit
    // fails at the worst moment: someone asking how to use the bot.
    #[test]
    fn help_texts_fit_in_a_telegram_message() {
        const LIMIT: usize = 4096;
        let mut pages = vec![("/help".to_string(), help_text())];
        for section in ["", "ml", "user", "group", "rules", "nonsense-falls-back-to-index"] {
            pages.push((format!("/help_op {section}"), help_op_text(section)));
        }
        for (name, text) in pages {
            let n = text.chars().count();
            assert!(n <= LIMIT, "{name} is {n} chars, over Telegram's {LIMIT} limit");
            assert!(!text.is_empty(), "{name} is empty");
            // Unclosed tags render as literal text and look broken.
            assert_eq!(text.matches("<b>").count(), text.matches("</b>").count(), "{name} has unbalanced <b>");
            assert_eq!(text.matches("<code>").count(), text.matches("</code>").count(), "{name} has unbalanced <code>");
            assert_eq!(text.matches("<i>").count(), text.matches("</i>").count(), "{name} has unbalanced <i>");
        }
    }

    // The first version of this rewrite looked fine in source and wrapped
    // badly on a phone: `<code>` renders monospace, monospace CJK is wide,
    // and ideographic spaces used as pseudo-columns don't align - a wrapped
    // line also loses its indent. Observed wrap point was ~43 columns, so
    // list lines are held well under that. Prose paragraphs are exempt:
    // they wrap at word boundaries and read correctly.
    #[test]
    fn help_list_lines_are_narrow_enough_for_a_phone() {
        const MAX_COLUMNS: usize = 38;
        let strip = |t: &str| {
            let (mut out, mut in_tag) = (String::new(), false);
            for c in t.chars() {
                match c {
                    '<' => in_tag = true,
                    '>' => in_tag = false,
                    _ if !in_tag => out.push(c),
                    _ => {}
                }
            }
            out.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        };
        // CJK glyphs occupy two columns.
        let width = |l: &str| -> usize { l.chars().map(|c| if (c as u32) > 0x2E80 { 2 } else { 1 }).sum() };

        for (name, text) in [
            ("/help", help_text()),
            ("/help_op", help_op_text("")),
            ("/help_op ml", help_op_text("ml")),
            ("/help_op user", help_op_text("user")),
            ("/help_op group", help_op_text("group")),
            ("/help_op rules", help_op_text("rules")),
        ] {
            let plain = strip(&text);
            assert!(!plain.contains('\u{3000}'), "{name} uses an ideographic space as a column separator");
            for line in plain.lines() {
                // A command/bullet line must not wrap. Prose is allowed to.
                let is_list_line = line.starts_with('/') || line.starts_with('·');
                if is_list_line {
                    assert!(width(line) <= MAX_COLUMNS, "{name}: line is {} columns, will wrap: {line}", width(line));
                }
            }
        }
    }

    // The public help must not advertise the maintainer-gated module, and
    // must still cover the ones groups are expected to turn on themselves.
    #[test]
    fn public_help_lists_every_public_module_and_hides_warn_pol() {
        let help = help_text();
        for (_, display, _) in PUBLIC_MODULES {
            assert!(help.contains(display), "/help never mentions the {display} module");
        }
        assert!(!help.to_lowercase().contains("warn-pol"), "warn-pol must stay out of the public help");
        assert!(!help_op_text("").to_lowercase().contains("/magic"), "/magic stays undocumented");
    }

    // A /spam confirmation must be findable when the review lands (possibly
    // after a restart), and consumed exactly once so a later reject can't
    // re-edit a message an approve already rewrote.
    #[tokio::test]
    async fn report_confirmation_is_stored_and_taken_once() {
        let runtime = test_runtime().await;
        assert!(runtime.take_report_confirmation("case-x").await.is_none());

        runtime.set_report_confirmation("case-x", -100, 42).await.unwrap();
        assert_eq!(runtime.take_report_confirmation("case-x").await, Some((-100, 42)));
        // Consumed on read - a second decision finds nothing to edit.
        assert!(runtime.take_report_confirmation("case-x").await.is_none());
    }

    // A /sb on a pure bot-mention message becomes a regex rule, not an ML
    // sample. The detection has to accept the real summon shapes (bare
    // handle, several handles, a trailing emoji) and reject anything with
    // real words or a non-bot mention.
    #[test]
    fn bot_mentions_only_matches_summons_and_rejects_conversation() {
        assert_eq!(bot_mentions_only("@iurfdervfbot"), Some(vec!["iurfdervfbot".to_string()]));
        assert_eq!(
            bot_mentions_only("@a_bot @b_bot 🎉"),
            Some(vec!["a_bot".to_string(), "b_bot".to_string()])
        );
        // Case-insensitive and de-duplicated.
        assert_eq!(bot_mentions_only("@SpamBot @spambot"), Some(vec!["spambot".to_string()]));

        // Real text alongside the mention -> not a bare summon.
        assert_eq!(bot_mentions_only("try @iurfdervfbot now"), None);
        assert_eq!(bot_mentions_only("看看 @somebot 這個"), None);
        // A mention that isn't a bot.
        assert_eq!(bot_mentions_only("@realuser"), None);
        // No mention at all.
        assert_eq!(bot_mentions_only("hello"), None);
        assert_eq!(bot_mentions_only("🎉🎉"), None);
    }

    // The host is a source constant, so its authority must hold with an
    // empty database and cannot be granted or revoked. Maintainers are
    // command-granted rows; a fresh database has none, which is the
    // "remove all current maintainers" outcome.
    #[tokio::test]
    async fn host_is_always_maintainer_and_cannot_be_removed_from_the_table() {
        let runtime = test_runtime().await;
        assert!(is_host(HOST_ID));
        assert!(runtime.is_maintainer(HOST_ID).await, "host must be a maintainer with no DB row at all");
        assert_eq!(runtime.list_maintainers().await.unwrap().len(), 0, "fresh DB has no granted maintainers");

        // Even a stray delete/insert of the host id doesn't change host status.
        runtime.set_maintainer(HOST_ID, false, Some(1)).await.unwrap();
        assert!(runtime.is_maintainer(HOST_ID).await, "host authority is not stored in the table");
    }

    // Maintainers are granted and revoked like reviewers, scoped per user,
    // and the grant survives a restart (cache is loaded from the table).
    #[tokio::test]
    async fn maintainer_grants_round_trip_and_persist() {
        let dir = std::env::temp_dir().join(format!("spb_test_{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let config = Config {
            bot_token: "test".to_string(),
            log_channel_id: -1,
            report_channel_id: -1,
            test_group_id: None,
            data_dir: dir.clone(),
            sqlite_path: dir.join("bot.db"),
            spam_threshold: 0.85,
            owner_id: None,
            hostctl_secret: None,
            hostctl_cmd: "/hostctl".to_string(),
        };
        {
            let runtime = Runtime::load(config.clone()).await.unwrap();
            assert!(!runtime.is_maintainer(555).await);
            runtime.set_maintainer(555, true, Some(HOST_ID)).await.unwrap();
            assert!(runtime.is_maintainer(555).await);
            assert!(!runtime.is_maintainer(666).await, "granting one must not grant another");
        }
        let restarted = Runtime::load(config).await.unwrap();
        assert!(restarted.is_maintainer(555).await, "a maintainer grant must survive a restart");

        restarted.set_maintainer(555, false, Some(HOST_ID)).await.unwrap();
        assert!(!restarted.is_maintainer(555).await);
    }

    // Backs /reviewer add|del|list and the report-channel button guard.
    // Granting must be scoped to the one account named, and revoking must
    // actually revoke - a stale grant here means someone keeps the ability
    // to approve network bans and write to the shared model.
    #[tokio::test]
    async fn reviewer_grants_round_trip_and_are_scoped_per_user() {
        let runtime = test_runtime().await;
        assert!(!runtime.is_reviewer(555).await);

        runtime.set_reviewer(555, true, Some(1)).await.unwrap();
        assert!(runtime.is_reviewer(555).await);
        assert!(!runtime.is_reviewer(666).await, "granting one user must not grant anyone else");

        // Re-granting is idempotent, not a duplicate row.
        runtime.set_reviewer(555, true, Some(1)).await.unwrap();
        assert_eq!(runtime.list_reviewers().await.unwrap().len(), 1);

        runtime.set_reviewer(555, false, Some(1)).await.unwrap();
        assert!(!runtime.is_reviewer(555).await);
        assert!(runtime.list_reviewers().await.unwrap().is_empty());
    }

    // /leave refuses non-group and project-owned chats. This matters more
    // than a normal input check: a blacklisted chat is refused before
    // command handling, so banning the project chat or log channel would
    // leave nowhere to type /forgive - and a bare /leave in a DM would
    // blacklist the maintainer's own chat id. Mirrors the handler's guard.
    #[tokio::test]
    async fn leave_target_guard_rejects_dms_and_project_infrastructure() {
        let runtime = test_runtime().await;
        runtime.set_project_chat(-1000).await;

        let protected = |target: i64, project: Option<i64>| {
            let mut list = vec![runtime.config.log_channel_id, runtime.config.report_channel_id];
            list.extend(project);
            target >= 0 || list.contains(&target)
        };
        let project = runtime.project_chat().await;
        assert_eq!(project, Some(-1000));

        // A DM's chat id is the user's own id - positive, and exactly what a
        // bare /leave in a DM would target.
        assert!(protected(555, project), "a private chat must never be blacklistable");
        assert!(protected(0, project));
        assert!(protected(-1000, project), "the project chat must be protected");
        assert!(protected(runtime.config.log_channel_id, project), "the log channel must be protected");
        assert!(protected(runtime.config.report_channel_id, project), "the report channel must be protected");
        // An ordinary group is still fair game.
        assert!(!protected(-1002222222222, project));
    }

    // The caches are populated at startup, so a ban written by a previous
    // process has to survive a restart - otherwise every restart quietly
    // re-admits everyone who was ever banned.
    #[tokio::test]
    async fn project_denial_lists_survive_a_restart() {
        let dir = std::env::temp_dir().join(format!("spb_test_{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let config = Config {
            bot_token: "test".to_string(),
            log_channel_id: -1,
            report_channel_id: -1,
            test_group_id: None,
            data_dir: dir.clone(),
            sqlite_path: dir.join("bot.db"),
            spam_threshold: 0.85,
            owner_id: None,
            hostctl_secret: None,
            hostctl_cmd: "/hostctl".to_string(),
        };
        {
            let runtime = Runtime::load(config.clone()).await.unwrap();
            runtime.set_group_banned(-100, true, "abuse", Some(1)).await.unwrap();
            runtime.set_user_banned(555, true, "abuse", Some(1)).await.unwrap();
        }
        let restarted = Runtime::load(config).await.unwrap();
        assert!(restarted.is_group_banned(-100).await, "a group ban must be reloaded from disk at startup");
        assert!(restarted.is_user_banned(555).await, "a user ban must be reloaded from disk at startup");
    }

    // The whole point of the eligibility rule: a group running a lowered
    // spam_threshold_override bans at its own bar locally, but must not be
    // able to push those below-bar bans onto every other group. Judged
    // against the global threshold, never the group's own - otherwise one
    // group setting a near-zero threshold turns every message it sees into
    // a project-wide ban.
    #[test]
    fn netban_eligibility_uses_the_global_bar_not_a_group_override() {
        let global = 0.85;

        // A group with a 0.01 override would ban on this locally; it must
        // not reach the shared blacklist.
        assert!(!netban_eligible(&ActionKind::AutoBan, Some(0.10), global, None));
        assert!(!netban_eligible(&ActionKind::AutoBan, Some(0.84), global, None));

        // Clears the project bar. Regex rule matches score 1.0, so
        // maintainer-managed rules land here too.
        assert!(netban_eligible(&ActionKind::AutoBan, Some(0.85), global, None));
        assert!(netban_eligible(&ActionKind::AutoBan, Some(1.0), global, None));
    }

    // The project/group dividing line: one group admin, or one group's house
    // rules, must not be able to impose a ban on every other group. Only a
    // project-level determination does that.
    #[test]
    fn netban_eligibility_excludes_group_level_decisions() {
        let global = 0.85;

        // A group admin's /sb is a call about their own room. Route it
        // through /spam if it deserves to be project-wide.
        assert!(!netban_eligible(&ActionKind::SpamBan, None, global, None));
        assert!(!netban_eligible(&ActionKind::SpamBan, Some(0.99), global, None));

        // NoHalal / NoLongName are unscored per-group policy opt-ins, not
        // spam findings - they arrive as AutoBan with no score.
        assert!(!netban_eligible(&ActionKind::AutoBan, None, global, None));

        // Not bans at all.
        assert!(!netban_eligible(&ActionKind::Mute, None, global, None));
        assert!(!netban_eligible(&ActionKind::Kick, None, global, None));
        assert!(!netban_eligible(&ActionKind::FloodMute, None, global, None));

        // Project-level determinations still qualify: a maintainer pressed
        // approve, or it's guest-mode abuse (inherently cross-group).
        assert!(netban_eligible(&ActionKind::ReportApproved, None, global, None));
        assert!(netban_eligible(&ActionKind::GuestBotBan, None, global, None));
        assert!(netban_eligible(&ActionKind::GuestInvokerBan, None, global, None));

        // BOTSPAM goes on the blacklist however it was caught - a manual /sb
        // (SpamBan) included, which is otherwise never eligible.
        assert!(netban_eligible(&ActionKind::SpamBan, None, global, Some("BOTSPAM")));
        assert!(netban_eligible(&ActionKind::AutoBan, Some(0.10), global, Some("BOTSPAM")));
        // A plain /sb with no BOTSPAM reason stays local.
        assert!(!netban_eligible(&ActionKind::SpamBan, None, global, Some("ML")));
    }

    // The flag must survive a re-persist: log_message_id backfills and status
    // changes rewrite the row through persist_case, which knows nothing about
    // netban - if it cleared the column, a user would silently drop off the
    // blacklist moments after landing on it.
    #[tokio::test]
    async fn netban_flag_survives_a_later_persist_case() {
        let runtime = test_runtime().await;
        let mut case = dummy_case(ActionKind::AutoBan, 100, 200, Utc::now());
        runtime.persist_case(&case).await.unwrap();
        runtime.mark_netban_eligible(&case.id).await.unwrap();

        case.log_message_id = Some(4242);
        case.status = "auto_banned".to_string();
        runtime.persist_case(&case).await.unwrap();

        assert_eq!(
            runtime.find_active_network_ban(200).await.unwrap().map(|c| c.id),
            Some(case.id.clone()),
            "re-persisting the case must not clear its netban flag"
        );
    }

    // The reported "unban then re-ban": /unban used to reverse only the
    // latest case in the current chat, so a second active ban elsewhere
    // kept the user on the blacklist and check_reban_and_act re-banned
    // them. find_active_bans_for_user must surface every active ban so the
    // handler can reverse them all.
    #[tokio::test]
    async fn unban_must_see_every_active_ban_not_just_one() {
        let runtime = test_runtime().await;
        let mut c1 = dummy_case(ActionKind::AutoBan, 100, 200, Utc::now() - chrono::TimeDelta::hours(1));
        c1.model_score = Some(0.99);
        let mut c2 = dummy_case(ActionKind::AutoBan, 300, 200, Utc::now());
        c2.model_score = Some(0.99);
        runtime.persist_case(&c1).await.unwrap();
        runtime.persist_case(&c2).await.unwrap();
        runtime.mark_netban_eligible(&c1.id).await.unwrap();
        runtime.mark_netban_eligible(&c2.id).await.unwrap();

        assert_eq!(runtime.find_active_bans_for_user(200).await.unwrap().len(), 2, "both bans must be visible");

        // Reversing only one leaves the user network-banned via the other -
        // exactly the old bug.
        let mut one = c2.clone();
        one.action = ActionKind::Unbanned;
        runtime.persist_case(&one).await.unwrap();
        assert!(runtime.find_active_network_ban(200).await.unwrap().is_some(), "still banned via the un-reversed case");
        assert!(runtime.find_active_ban_in_chat(100, 200).await.unwrap().is_some(), "reban net would still fire in group 100");

        // Reversing all of them clears the blacklist and the reban net.
        let mut two = c1.clone();
        two.action = ActionKind::Unbanned;
        runtime.persist_case(&two).await.unwrap();
        assert!(runtime.find_active_network_ban(200).await.unwrap().is_none());
        assert!(runtime.find_active_ban_in_chat(100, 200).await.unwrap().is_none());
    }

    // guest-mode bans must also count as "banned somewhere" so /unban
    // reverses them and they clear from the blacklist.
    #[tokio::test]
    async fn find_active_bans_includes_guest_bans() {
        let runtime = test_runtime().await;
        let g = dummy_case(ActionKind::GuestBotBan, 100, 200, Utc::now());
        runtime.persist_case(&g).await.unwrap();
        assert_eq!(runtime.find_active_bans_for_user(200).await.unwrap().len(), 1);
    }

    // Backs check_reban_and_act's same-chat ban-evasion safety net. Unlike
    // find_active_network_ban, this must NOT require netban - that flag only
    // controls cross-group propagation, not whether a chat remembers its own
    // past bans - and must be scoped strictly per-chat (a ban in chat 100
    // must not match a lookup in chat 300, even for the same user).
    #[tokio::test]
    async fn find_active_ban_in_chat_ignores_netban_and_is_scoped_per_chat() {
        let runtime = test_runtime().await;
        let case = dummy_case(ActionKind::AutoBan, 100, 200, Utc::now());
        runtime.persist_case(&case).await.unwrap();

        assert!(
            runtime.find_active_ban_in_chat(100, 200).await.unwrap().is_some(),
            "netban being off must not hide a same-chat ban"
        );
        assert!(
            runtime.find_active_ban_in_chat(300, 200).await.unwrap().is_none(),
            "a ban in chat 100 must not count as active in an unrelated chat 300"
        );
        assert!(runtime.find_active_ban_in_chat(100, 999).await.unwrap().is_none());
    }

    // Backs check_guest_bot_and_act's invoker correlation: a bare
    // `@botusername` mention (how guest mode is actually invoked) must
    // match case-insensitively and return the most recent one, but a longer
    // message that merely references the bot in passing must not - that's
    // ordinary conversation, not a guest-mode summon, and banning over it
    // would be a real false positive.
    #[tokio::test]
    async fn find_recent_guest_invoker_matches_bare_mention_not_ordinary_conversation() {
        let runtime = test_runtime().await;
        runtime.record_recent_message(100, 111, MessageId(1), "Alice", "hello everyone").await;
        runtime.record_recent_message(100, 222, MessageId(2), "Bob", "hey has anyone actually used @iurfdervfbot before, is it any good?").await;
        runtime.record_recent_message(100, 333, MessageId(3), "Carol", "@IURFdervfBOT").await;

        let found = runtime.find_recent_guest_invoker(100, "iurfdervfbot").await;
        assert_eq!(found.map(|(id, _, _, _)| id), Some(333), "must match case-insensitively and prefer the most recent bare mention, skipping Bob's ordinary sentence");

        assert!(
            runtime.find_recent_guest_invoker(200, "iurfdervfbot").await.is_none(),
            "must be scoped per chat - nothing was recorded in chat 200"
        );
        assert!(
            runtime.find_recent_guest_invoker(100, "someotherbot").await.is_none(),
            "must not match a bot username nobody actually mentioned"
        );
    }

    // Regression check for a reported evasion: spammers summon several guest
    // bots in one message and tack an emoji on the end. Stripping only the
    // bot being checked left the other handles in the remainder, so every one
    // of the three looked like ordinary chatter and nobody got banned. Each
    // summoned bot must independently resolve back to the same invoker.
    #[tokio::test]
    async fn find_recent_guest_invoker_catches_multi_bot_summon_with_emoji() {
        let runtime = test_runtime().await;
        runtime.record_recent_message(100, 444, MessageId(9), "Spammer", "@botone @bottwo @botthree 🎉").await;

        for username in ["botone", "bottwo", "botthree"] {
            assert_eq!(
                runtime.find_recent_guest_invoker(100, username).await.map(|(id, _, _, _)| id),
                Some(444),
                "{username} was summoned in that message, so it must resolve back to the invoker"
            );
        }
    }

    // forget_recent_message backs the "one summon, one ban" guarantee: with
    // three guest bots replying to a single summon, only the first reply may
    // find the invoker - otherwise each opens a duplicate case.
    #[tokio::test]
    async fn forget_recent_message_stops_repeat_matches_for_one_summon() {
        let runtime = test_runtime().await;
        runtime.record_recent_message(100, 444, MessageId(9), "Spammer", "@botone @bottwo 🎉").await;

        assert!(runtime.find_recent_guest_invoker(100, "botone").await.is_some());
        runtime.forget_recent_message(100, MessageId(9)).await;
        assert!(
            runtime.find_recent_guest_invoker(100, "bottwo").await.is_none(),
            "the second guest reply must find nothing once the summon has been acted on"
        );
    }

    // strip_mentions has to end each handle at the first non-username
    // character, so CJK text and emoji survive into the remainder - that
    // remainder is exactly what disqualifies a message from being a summon.
    #[test]
    fn strip_mentions_removes_handles_and_keeps_everything_else() {
        assert_eq!(strip_mentions("@botone @bottwo 🎉").trim(), "🎉");
        assert_eq!(strip_mentions("@bot_one_2").trim(), "");
        assert_eq!(strip_mentions("看看 @somebot 這個").trim(), "看看  這個".trim());
        assert_eq!(strip_mentions("no mentions here"), "no mentions here");
    }

    // Regression check for the 777000 ("Telegram") incident: the reban
    // safety net kept deleting/re-banning routine linked-channel-forward
    // announcements because it didn't know 777000 isn't a real, bannable
    // account. Every id on the exemption list must be recognized, and an
    // ordinary user id must not accidentally match.
    #[test]
    fn platform_pseudo_user_ids_are_recognized() {
        assert!(is_platform_pseudo_user(777000), "777000 = Telegram service/linked-channel-forward pseudo-account");
        assert!(is_platform_pseudo_user(1087968824), "GroupAnonymousBot");
        assert!(is_platform_pseudo_user(136817688), "Channel (anonymous channel post pseudo-sender)");
        assert!(!is_platform_pseudo_user(200), "an ordinary user id must not be treated as a platform pseudo-account");
    }

    // Same "reversal mutates action in place" property as
    // load_latest_case_by_actions - once reversed, it must stop being an
    // active network ban.
    #[tokio::test]
    async fn find_active_network_ban_excludes_reversed_case() {
        let runtime = test_runtime().await;
        let mut case = dummy_case(ActionKind::SpamBan, 100, 200, Utc::now());
        runtime.persist_case(&case).await.unwrap();
        runtime.mark_netban_eligible(&case.id).await.unwrap();
        assert!(runtime.find_active_network_ban(200).await.unwrap().is_some());

        case.action = ActionKind::Unbanned;
        runtime.persist_case(&case).await.unwrap();
        assert!(runtime.find_active_network_ban(200).await.unwrap().is_none());
    }

    // Backs the PM appeal bridge's "is this user banned anywhere, and
    // where/why" query: unlike find_active_network_ban, this must see bans
    // in every group, not just netban-participating ones, and return every
    // independent active ban across distinct chats, not just the latest.
    #[tokio::test]
    async fn find_active_bans_for_user_spans_all_chats_without_netban() {
        let runtime = test_runtime().await;
        let case_a = dummy_case(ActionKind::AutoBan, 100, 200, Utc::now() - chrono::TimeDelta::hours(1));
        let case_b = dummy_case(ActionKind::SpamBan, 300, 200, Utc::now());
        let unrelated_user = dummy_case(ActionKind::AutoBan, 100, 999, Utc::now());
        runtime.persist_case(&case_a).await.unwrap();
        runtime.persist_case(&case_b).await.unwrap();
        runtime.persist_case(&unrelated_user).await.unwrap();

        let mut found: Vec<String> = runtime.find_active_bans_for_user(200).await.unwrap().into_iter().map(|c| c.id).collect();
        found.sort();
        let mut expected = vec![case_a.id.clone(), case_b.id.clone()];
        expected.sort();
        assert_eq!(found, expected);
    }

    #[tokio::test]
    async fn find_active_bans_for_user_excludes_reversed_case() {
        let runtime = test_runtime().await;
        let mut case = dummy_case(ActionKind::AutoBan, 100, 200, Utc::now());
        runtime.persist_case(&case).await.unwrap();
        assert_eq!(runtime.find_active_bans_for_user(200).await.unwrap().len(), 1);

        case.action = ActionKind::Unbanned;
        runtime.persist_case(&case).await.unwrap();
        assert!(runtime.find_active_bans_for_user(200).await.unwrap().is_empty());
    }

    // strike_count means "was ever banned," not "is currently banned" - it
    // must count a since-reversed ban too, not just active ones.
    #[tokio::test]
    async fn count_ban_strikes_counts_active_and_reversed_bans() {
        let runtime = test_runtime().await;
        let active = dummy_case(ActionKind::AutoBan, 100, 200, Utc::now());
        runtime.persist_case(&active).await.unwrap();
        let mut reversed = dummy_case(ActionKind::SpamBan, 300, 200, Utc::now());
        runtime.persist_case(&reversed).await.unwrap();
        reversed.action = ActionKind::Unbanned;
        runtime.persist_case(&reversed).await.unwrap();

        assert_eq!(runtime.count_ban_strikes_for_user(200).await.unwrap(), 2);
        assert_eq!(runtime.count_ban_strikes_for_user(999).await.unwrap(), 0);
    }

    #[test]
    fn parse_exchange_envelope_accepts_plain_and_code_fenced_json() {
        let plain = r#"{"from":"PM","to":["GBB"],"action":"query","type":"bad","data":{"id":1}}"#;
        let envelope = parse_exchange_envelope(plain).expect("plain JSON should parse");
        assert_eq!(envelope.from, "PM");
        assert_eq!(envelope.action, "query");
        assert_eq!(envelope.kind, "bad");

        let fenced = format!("```json\n{plain}\n```");
        let envelope = parse_exchange_envelope(&fenced).expect("code-fenced JSON should parse");
        assert_eq!(envelope.from, "PM");

        assert!(parse_exchange_envelope("not json at all").is_none());
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

    // Backs /magic: a chat starts un-allowed for any module, allowing it
    // flips is_module_allowed, and disallowing removes it again - and this
    // must be scoped per (module, chat_id), not global.
    #[tokio::test]
    async fn module_allowlist_round_trip_and_scoped_per_module() {
        let runtime = test_runtime().await;
        assert!(!runtime.is_module_allowed("warn-pol", 100).await.unwrap());

        runtime.set_module_allowed("warn-pol", 100, true, Some(1)).await.unwrap();
        assert!(runtime.is_module_allowed("warn-pol", 100).await.unwrap());
        // A different module, or a different chat, must not be affected.
        assert!(!runtime.is_module_allowed("some-other-module", 100).await.unwrap());
        assert!(!runtime.is_module_allowed("warn-pol", 200).await.unwrap());

        runtime.set_module_allowed("warn-pol", 100, false, Some(1)).await.unwrap();
        assert!(!runtime.is_module_allowed("warn-pol", 100).await.unwrap());
    }

    // Backs /pol's warn-then-ban escalation: a fresh (chat, user) pair has
    // no warnings, incrementing accumulates a real count (not just a
    // boolean flag), and /pol clear resets it back to zero.
    #[tokio::test]
    async fn pol_warnings_increment_and_clear() {
        let runtime = test_runtime().await;
        assert_eq!(runtime.pol_warn_count(100, 200).await.unwrap(), 0);

        let after_first = runtime.increment_pol_warn(100, 200).await.unwrap();
        assert_eq!(after_first, 1);
        assert_eq!(runtime.pol_warn_count(100, 200).await.unwrap(), 1);

        let after_second = runtime.increment_pol_warn(100, 200).await.unwrap();
        assert_eq!(after_second, 2);

        // A different chat's count for the same user must be independent -
        // warns are per-group, not global.
        assert_eq!(runtime.pol_warn_count(999, 200).await.unwrap(), 0);

        runtime.clear_pol_warns(100, 200).await.unwrap();
        assert_eq!(runtime.pol_warn_count(100, 200).await.unwrap(), 0);
    }

    // Backs /module warn-pol on|off: the pol flag round-trips through
    // get_group_modules/set_group_module exactly like every other module.
    #[tokio::test]
    async fn warn_pol_module_flag_round_trips() {
        let runtime = test_runtime().await;
        assert!(!runtime.get_group_modules(100).await.unwrap().pol);
        runtime.set_group_module(100, "warn-pol", true).await.unwrap();
        assert!(runtime.get_group_modules(100).await.unwrap().pol);
        runtime.set_group_module(100, "warn-pol", false).await.unwrap();
        assert!(!runtime.get_group_modules(100).await.unwrap().pol);
    }

    // Backs check_guest_bot_and_act / /module guestban on|off: unlike every
    // other opt-in content-policy module, guest_ban must default to true for
    // a brand-new chat (baseline hygiene against an actively-exploited
    // Telegram feature, matching flood_control's own default-on precedent),
    // and still round-trips through get_group_modules/set_group_module like
    // any other flag.
    #[tokio::test]
    async fn guest_ban_module_flag_defaults_on_and_toggles() {
        let runtime = test_runtime().await;
        assert!(runtime.get_group_modules(100).await.unwrap().guest_ban);
        runtime.set_group_module(100, "guestban", false).await.unwrap();
        assert!(!runtime.get_group_modules(100).await.unwrap().guest_ban);
        runtime.set_group_module(100, "guestban", true).await.unwrap();
        assert!(runtime.get_group_modules(100).await.unwrap().guest_ban);
    }

    // PUBLIC_MODULES drives both the status listing and `/module all`, so
    // every key in it must be a real module that set_group_module actually
    // writes - a typo would silently list/toggle nothing. warn-pol must stay
    // out of it: it's maintainer-gated and must never appear in a public
    // list or get flipped on by a bulk toggle.
    #[tokio::test]
    async fn public_modules_table_is_settable_and_excludes_warn_pol() {
        let runtime = test_runtime().await;
        assert!(!PUBLIC_MODULES.iter().any(|(key, _, _)| *key == "warn-pol"));

        for (key, _, _) in PUBLIC_MODULES {
            runtime.set_group_module(100, key, true).await.unwrap();
        }
        let all_on = runtime.get_group_modules(100).await.unwrap();
        for (key, _, _) in PUBLIC_MODULES {
            assert_eq!(module_flag(&all_on, key), Some(true), "{key} should be on");
        }
        assert!(!all_on.pol, "a bulk enable must not touch warn-pol");

        for (key, _, _) in PUBLIC_MODULES {
            runtime.set_group_module(100, key, false).await.unwrap();
        }
        let all_off = runtime.get_group_modules(100).await.unwrap();
        for (key, _, _) in PUBLIC_MODULES {
            assert_eq!(module_flag(&all_off, key), Some(false), "{key} should be off");
        }
    }

    // `/module all off` must leave the baseline protections alone, so a
    // group can't strip its own spam defences in one keystroke. The
    // baseline flags must be exactly the ones GroupModuleSettings defaults
    // to on - if a future module changes its default, this catches the
    // table drifting out of sync with it.
    #[test]
    fn baseline_modules_match_the_default_on_flags() {
        let defaults = GroupModuleSettings::default();
        for (key, display, baseline) in PUBLIC_MODULES {
            assert_eq!(
                module_flag(&defaults, key),
                Some(*baseline),
                "{display} is marked baseline={baseline} but its default says otherwise"
            );
        }
        assert!(
            PUBLIC_MODULES.iter().any(|(_, _, baseline)| *baseline),
            "at least one module must be baseline, or the skip logic is dead code"
        );
    }

    // Backs /ml_dedupe: simulates the real dirty-data shape (a spam phrase
    // mass-imported 3 times, an empty-text sample, and one clean distinct
    // sample of each label) and confirms dedup collapses duplicates to one
    // copy, drops the empty sample entirely, and leaves distinct samples
    // and doc/token counts consistent with what's actually left.
    #[tokio::test]
    async fn dedupe_training_samples_collapses_duplicates_and_drops_empty() {
        let runtime = test_runtime().await;
        train_spam(&runtime, "重複垃圾訊息", None).await.unwrap();
        train_spam(&runtime, "重複垃圾訊息", None).await.unwrap();
        train_spam(&runtime, "重複垃圾訊息", None).await.unwrap();
        train_spam(&runtime, "獨特垃圾訊息", None).await.unwrap();
        train_spam(&runtime, "", None).await.unwrap();
        train_ham(&runtime, "正常聊天內容", None).await.unwrap();

        let (dup_removed, empty_removed) = runtime.dedupe_training_samples().await.unwrap();
        assert_eq!(dup_removed, 2, "3 copies of the same text should collapse to 1, removing 2");
        assert_eq!(empty_removed, 1);

        let rebuilt = runtime.rebuild_model().await.unwrap();
        assert_eq!(rebuilt.spam_docs, 2, "one surviving copy of the duplicate + the distinct sample");
        assert_eq!(rebuilt.ham_docs, 1);

        let remaining: i64 = runtime
            .with_conn(|conn| Ok(conn.query_row("SELECT COUNT(*) FROM training_samples WHERE label = 'spam' AND text = '重複垃圾訊息'", [], |row| row.get(0))?))
            .await
            .unwrap();
        assert_eq!(remaining, 1);

        // Re-running must be a no-op - nothing left to deduplicate.
        let (dup_removed_again, empty_removed_again) = runtime.dedupe_training_samples().await.unwrap();
        assert_eq!((dup_removed_again, empty_removed_again), (0, 0));
    }

    // Regression test for the /spam-mistargets-topic-creator bug: every
    // message sent inside a forum topic carries a synthetic
    // reply_to_message pointing at that topic's own creation service
    // message (real payload shape confirmed against teloxide-core's own
    // "topic_message" test fixture) - its `from` is whoever created the
    // topic, not whoever's currently posting, and real_reply must not treat
    // it as a genuine reply.
    #[test]
    fn real_reply_ignores_synthetic_forum_topic_root() {
        let topic_root_reply_json = r#"{"chat":{"id":-1001847508954,"is_forum":true,"title":"twest","type":"supergroup"},"date":1675229140,"from":{"first_name":"вафель'","id":1253681278,"is_bot":false,"language_code":"en","username":"wafflelapkin"},"is_topic_message":true,"message_id":5,"message_thread_id":4,"reply_to_message":{"chat":{"id":-1001847508954,"is_forum":true,"title":"twest","type":"supergroup"},"date":1675229139,"forum_topic_created":{"icon_color":9367192,"icon_custom_emoji_id":"5312536423851630001","name":"???"},"from":{"first_name":"вафель'","id":1253681278,"is_bot":false,"language_code":"en","username":"wafflelapkin"},"is_topic_message":true,"message_id":4,"message_thread_id":4},"text":"/spam"}"#;
        let message: Message = serde_json::from_str(topic_root_reply_json).unwrap();

        assert!(message.reply_to_message().is_some(), "Telegram really does attach a reply_to_message here");
        assert!(
            message.reply_to_message().unwrap().forum_topic_created().is_some(),
            "and it's the topic-creation service message, not a real reply"
        );
        assert!(real_reply(&message).is_none(), "real_reply must filter this out");

        // A genuine reply (same shape, but the replied-to message is an
        // ordinary text message, not a topic-creation service message)
        // must still work normally.
        let genuine_reply_json = r#"{"chat":{"id":-1001847508954,"is_forum":true,"title":"twest","type":"supergroup"},"date":1675229140,"from":{"first_name":"вафель'","id":1253681278,"is_bot":false,"language_code":"en","username":"wafflelapkin"},"is_topic_message":true,"message_id":6,"message_thread_id":4,"reply_to_message":{"chat":{"id":-1001847508954,"is_forum":true,"title":"twest","type":"supergroup"},"date":1675229139,"from":{"first_name":"SpammerBot","id":999,"is_bot":false},"is_topic_message":true,"message_id":5,"message_thread_id":4,"text":"buy now"},"text":"/sb"}"#;
        let message: Message = serde_json::from_str(genuine_reply_json).unwrap();
        let reply = real_reply(&message).expect("a genuine reply must still be returned");
        assert_eq!(reply.from.as_ref().unwrap().id.0, 999);
    }
}





