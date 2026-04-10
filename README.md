# Not So Smart Anti-Spam Bot

[English](#english) | [繁體中文](#chinese)

<a name="english"></a>

An asynchronous Telegram anti-spam bot built with Rust, featuring a Naive Bayes classifier, manual moderation tools, and a feedback loop for model training.

## Tech Stack

- Naive Bayes text classifier for spam detection
- `teloxide` for Telegram Bot API interactions
- `tokio` for the asynchronous runtime
- SQLite (via `rusqlite`) for persisting training samples and case metadata
- `serde`, `chrono`, and `uuid` for data serialization, timestamping, and case identification

## Features

- Evaluates message content and user display names to calculate a spam score.
- Automatically deletes messages and bans users when the score exceeds a configurable threshold.
- Logs all automated and manual actions to a dedicated channel with a traceable case ID.
- Allows group admins to use `/sb` (Spam Ban), `/mute`, and `/kick` via replies.
- Enables users to report suspicious content using `/spam`. Reports are routed to a private channel for maintainer review.
- Provides a `/case <id>` command for anyone to look up the details of a specific moderation action.
- Allows maintainers to refine the model using `/ml_train_spam` and `/ml_clean_spam` to fix false positives/negatives.
- Includes tools for mass training from raw logs, rebuilding the model, purging specific samples, and adjusting sensitivity thresholds dynamically.

## Configuration

Set the following environment variables before running:

| Variable | Description |
| :--- | :--- |
| `BOT_TOKEN` | Telegram Bot Token. |
| `LOG_CHANNEL_ID` | Channel ID for moderation logs. |
| `REPORT_CHANNEL_ID` | Private channel ID for reviewing user reports. |
| `MAINTAINER_IDS` | Comma-separated User IDs allowed to manage the model. |
| `SQLITE_PATH` | Path to the SQLite database (default: `data/bot.db`). |
| `SPAM_THRESHOLD` | Score threshold for auto-ban (default: `0.85`). |

## Quick Start

1. Install the Rust toolchain.
2. Configure the environment variables:
   ```bash
   export BOT_TOKEN="your_token"
   export LOG_CHANNEL_ID="-100..."
   export REPORT_CHANNEL_ID="-100..."
   export MAINTAINER_IDS="123,456"
   ```
3. Build and run:
   ```bash
   cargo run --release
   ```

---

<a name="chinese"></a>

# Not So Smart Anti-Spam Bot（不是很聰明的反廣告機器人）

使用 Rust 開發的 Telegram 反廣告機器人。結合 Naive Bayes 分類器、管理員工具以及可回溯的模型訓練機制。

- 使用 Naive Bayes 演算法進行字詞分類與 Spam 偵測
- `teloxide` 處理 Telegram Bot API 交互
- `tokio` 作為異步執行環境
- SQLite (`rusqlite`) 儲存訓練樣本與案件資料
- `serde`、`chrono` 與 `uuid` 處理資料序列化、時間紀錄及案件編號

## 功能

- 同時將訊息正文與發言者顯示名稱（Display Name）納入評分特徵。
- 當 Spam 分數超過設定門檻時，自動刪除訊息並封禁使用者。
- 所有自動與手動操作皆寫入日誌頻道，並生成唯一的 case ID。
- 管理員可透過回覆訊息執行 `/sb` (封禁並寫入模型)、`/mute` (禁言) 與 `/kick` (踢出)。
- 使用者可透過 `/spam` 舉報訊息，交由維護組在私有頻道進行按鈕審核。
- 群組管理員與維護人員不會在服務群被封禁。
- 只有 `TEST_GROUP_ID` 內的訊息只做評分、不封禁。
- 提供 `/case <id>` 指令供任何人查詢特定案件的處置細節。
- 提供 `/ml_score` 指令測試單條文本分數，僅 maintainer 可用。
- 維護組可使用 `/ml_train_spam`、`/ml_clean_spam` 與 `/mark_ham` 即時修正模型。
- 支援從原始日誌中批量抽取正文訓練、重建模型、清除特定樣本以及動態調整封禁門檻。
- 支援 `/ml_start_mass_ham` 與 `/ml_finish_mass_ham` 批量標記 ham。
- `TEST_GROUP_ID` 內每條消息都會回傳分數，但不會觸發封禁。
- 項目組指令僅 maintainer 可見。
- `/ml_score_debug` 與 `/ml_score` 僅 maintainer 可用。

## 環境變數

啟動前需設定以下環境變數：

| 變數名 | 說明 |
| :--- | :--- |
| `BOT_TOKEN` | Telegram 機器人 Token。 |
| `LOG_CHANNEL_ID` | 操作日誌頻道 ID。 |
| `REPORT_CHANNEL_ID` | 舉報審核頻道 ID。 |
| `TEST_GROUP_ID` | 測試群組 ID，所有消息只做評分不封禁。 |
| `MAINTAINER_IDS` | 維護人員 User ID (多個以逗號分隔)。 |
| `SQLITE_PATH` | 資料庫路徑 (預設 `data/bot.db`)。 |
| `SPAM_THRESHOLD` | 自動封禁門檻 (預設 `0.85`)。 |

## 快速啟動

1. 安裝 Rust 工具鏈。
2. 設定環境變數：
   ```bash
   export BOT_TOKEN="你的Token"
    export LOG_CHANNEL_ID="-100..."
    export REPORT_CHANNEL_ID="-100..."
    export TEST_GROUP_ID="-100..."
    export MAINTAINER_IDS="123,456"
   ```
3. 編譯並執行：
   ```bash
   cargo run --release
   ```
