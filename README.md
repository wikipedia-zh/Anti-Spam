# Not So Smart Anti-Spam Bot

An asynchronous Telegram anti-spam bot built with Rust, featuring a Naive Bayes classifier, manual moderation tools, and a feedback loop for model training.

## Tech Stack

- Naive Bayes text classifier for spam detection
- `teloxide` for Telegram Bot API interactions
- `tokio` for the asynchronous runtime
- SQLite (via `rusqlite`) for persisting training samples and case metadata
- `serde`, `chrono`, and `uuid` for data serialization, timestamping, and case identification
- `anyhow` for error context and propagation

## Features

- Evaluates message content and user display names to calculate a spam score.
- Automatically deletes messages and bans users when the score exceeds a configurable threshold.
- Logs all automated and manual actions to a dedicated channel with a traceable case ID.
- Allows group admins to use `/sb` (Spam Ban), `/mute`, and `/kick` via replies.
- Prevents `/sb` from being used against group admins.
- Enables users to report suspicious content using `/spam` or `/report`. Reports are routed to a private review channel.
- Provides a `/case <id>` command for anyone to look up the details of a specific moderation action.
- Allows maintainers to refine the model using `/ml_train_spam` and `/ml_clean_spam` to fix false positives/negatives.
- Includes tools for mass training from raw logs, rebuilding the model, purging specific samples, and adjusting sensitivity thresholds dynamically.

## Configuration

Set the following environment variables before running:

| Variable | Description |
| :--- | :--- |
| `BOT_TOKEN` | Telegram Bot Token. |
| `LOG_CHANNEL_ID` | Channel ID for public moderation logs. |
| `REPORT_CHANNEL_ID` | Private channel ID for reviewing user reports. |
| `MAINTAINER_IDS` | Comma-separated User IDs allowed to manage the model. |
| `SQLITE_PATH` | Path to the SQLite database (default: `data/bot.db`). |
| `SPAM_THRESHOLD` | Score threshold for auto-ban (default: `0.85`). |
| `MIN_TRAINING_SAMPLES` | Minimum sample count before threshold tuning (default: `20`). |

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

## 中文

# Not So Smart Anti-Spam Bot（不是很聰明的反廣告機器人）

使用 Rust 開發的異步 Telegram 反廣告機器人。結合 Naive Bayes 分類器、管理員工具以及可回溯的模型訓練機制。

## 技術棧

- 使用 Naive Bayes 演算法進行字詞分類與 Spam 偵測
- `teloxide` 處理 Telegram Bot API 交互
- `tokio` 作為異步執行環境
- SQLite (`rusqlite`) 儲存訓練樣本與案件資料
- `serde`、`chrono` 與 `uuid` 處理資料序列化、時間紀錄及案件編號
- `anyhow` 處理錯誤上下文

## 功能

- 同時將訊息正文與發言者顯示名稱（Display Name）納入評分特徵。
- 當 Spam 分數超過設定門檻時，自動刪除訊息並封禁使用者。
- 所有自動與手動操作皆寫入公開日誌頻道，並生成唯一的 case ID。
- 管理員可透過回覆訊息執行 `/sb` (封禁並寫入模型)、`/mute` (禁言) 與 `/kick` (踢出)。
- `/sb` 不允許對群組管理員執行。
- 使用者可透過 `/spam` 或 `/report` 舉報訊息，交由私有舉報處理頻道進行按鈕審核。
- 提供 `/case <id>` 指令供任何人查詢特定案件的處置細節。
- 維護組可使用 `/ml_train_spam` 與 `/ml_clean_spam` 即時修正模型。
- 支援從原始日誌中批量抽取正文訓練、重建模型、清除特定樣本以及動態調整封禁門檻。

## 指令

### 群組管理員

- `/spamban` 或 `/sb`：回覆一條訊息後使用，封禁並踢出，並把內容寫入 spam 模型
- `/mute` 或 `/m`：回覆一條訊息後使用，禁言
- `/kick` 或 `/k`：回覆一條訊息後使用，踢出

### 所有人

- `/spam`：回覆一條疑似 spam 訊息，提交給維護組審核
- `/report`：`/spam` 別名
- `/case <id>`：查詢案例
- `/ml_score <文本>`：測試一段文本的 spam 分數
- `/ml_score`：回覆一條消息也可以直接測分數

### 項目維護組

- `/ml_train_spam`：回覆樣本，寫入 spam 模型
- `/ml_clean_spam`：回覆樣本，寫入 ham 模型，修正誤報
- `/ml_purge <case_id>`：刪除某個 case 對應的訓練樣本並重建模型
- `/ml_purge_text <文字片段>`：依文字片段清除誤訓練樣本並重建模型
- `/ml_rebuild`：用 SQLite 中的訓練樣本重建模型
- `/ml_stats`：查看樣本數與有效門檻
- `/ml_threshold <0.50-0.99>`：設定自動封禁門檻（請謹慎調整）
- `/ml_export`：匯出訓練資料
- `/ml_start_mass_train`：在私訊中啟動批量訓練收集模式
- `/ml_start_mass_train_smart`：啟動 smart 模式批量訓練
- `/ml_start_mass_train_plain`：啟動 plain 模式批量訓練
- `/ml_finish_mass_train`：結束收集並將累積的文本批量寫入模型
- `/ml_debug_parse`：回覆一段日誌，測試 smart 抽取結果

## 工作流程

1. 使用者在群組發言。
2. 機器人對非管理員訊息做 spam 評分。
3. 當分數高於有效門檻時，機器人刪訊息、封禁該使用者，並記錄到日誌頻道與群組。
4. 管理員使用 `/sb` 時，會直接執行並寫入模型。
5. 一般使用者使用 `/spam` 時，只會進入維護組審核，不會直接污染模型。
6. 維護組可透過按鈕選擇受理或拒絕。
7. 受理後寫入 spam 樣本，拒絕後寫入 ham 樣本。

## 日誌

- 所有動作都寫入 `LOG_CHANNEL_ID`
- 公開查詢一律使用 Telegram `t.me/c/...` 形式
- `/spam` 與 `/report` 會進入 `REPORT_CHANNEL_ID` 進行處理
- 舉報處理頻道是私有頻道，進入頻道者視為項目組成員

## SQLite

預設資料庫位置：`data/bot.db`

表格：

- `cases`：操作與審核紀錄
- `training_samples`：模型訓練資料
- `token_counts`：token 統計
- `model_meta`：模型元資料

## 環境變數

啟動前需設定以下環境變數：

| 變數名 | 說明 |
| :--- | :--- |
| `BOT_TOKEN` | Telegram 機器人 Token。 |
| `LOG_CHANNEL_ID` | 操作日誌頻道 ID。 |
| `REPORT_CHANNEL_ID` | 舉報審核頻道 ID。 |
| `MAINTAINER_IDS` | 維護人員 User ID (多個以逗號分隔)。 |
| `SQLITE_PATH` | 資料庫路徑 (預設 `data/bot.db`)。 |
| `SPAM_THRESHOLD` | 自動封禁門檻 (預設 `0.85`)。 |
| `MIN_TRAINING_SAMPLES` | 門檻自動微調前的最小樣本數 (預設 `20`)。 |

## 快速啟動

1. 安裝 Rust 工具鏈。
2. 設定環境變數：
   ```bash
   export BOT_TOKEN="你的Token"
   export LOG_CHANNEL_ID="-100..."
   export REPORT_CHANNEL_ID="-100..."
   export MAINTAINER_IDS="123,456"
   ```
3. 編譯並執行：
   ```bash
   cargo run --release
   ```

## 模型說明

- 目前使用的是 Naive Bayes 風格的字詞統計分類器。
- 模型由 token 頻率、spam/ham 先驗與平滑機制組成。
- 不使用深度學習或外部 ML 框架。

## 模型治理建議

1. 先用 `/ml_stats` 看樣本是否足夠。
2. 誤報多時，先用 `/ml_clean_spam` 或 `/ml_purge <case_id>` 清除錯誤樣本。
3. 大量調整後執行 `/ml_rebuild`。
4. 用 `/ml_threshold <值>` 微調敏感度。
5. 保持 spam 與 ham 樣本都持續補充，避免模型偏向單邊。
6. 若要快速灌大量樣本，請先在私訊中使用 `/ml_start_mass_train_smart`，然後直接貼原始日誌或純文本樣本，最後用 `/ml_finish_mass_train`。

## 故障排查

- 機器人不回應：確認 `BOT_TOKEN` 正確，且 bot 已加入群組
- 無法封禁：確認 bot 在群組有管理員權限
- 無法寫日誌：確認 `LOG_CHANNEL_ID` 正確，且 bot 在日誌頻道有發訊權限
- 無法審核舉報：確認 `REPORT_CHANNEL_ID` 正確，且 bot 在該私有頻道有發訊權限
- `/case` 查不到：確認該 case 已寫入 SQLite
- 誤封偏多：提高 `SPAM_THRESHOLD`，並執行 `/ml_clean_spam` 或 `/ml_rebuild`

## 備註

- `/spamban` 會自動把樣本寫入 spam 模型
- `/spam` 避免自動污染模型，必須由維護人員審核
