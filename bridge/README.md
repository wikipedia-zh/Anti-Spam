# hexDrill 同步

用專用 Telegram 使用者帳號讀取 SPB 資料庫，將符合條件的封禁送到指定私密群組的 `@hexdrill_bot`，只操作 `+spb` 清單。預設不發送。

收錄成功封禁、尚未撤銷的案件：模型分數按目前日誌四位小數顯示為 `1.0000`、命中 REGEX，或經項目人員核准的舉報。一般 `/sb`、訪客機器人及頻道身分不在範圍內。白名單和項目人員也會排除。舊舉報缺少核准紀錄時，只列入待核對清單。

## 設定

在 Linux 上使用 Python 3.11 以上版本，另建虛擬環境並安裝 `requirements.txt`。將 `config.example.json` 複製到私有目錄，再填入資料庫路徑和接收群組 ID。目錄權限設為 `700`，登入資料與 session 檔案設為 `600`。

`credentials_file` 指向包含 `api_id` 和 `api_hash` 的 JSON 檔案。請直接在主機上填寫，不要提交到 Git，也不要將驗證碼或 session 貼到聊天中。

先產生歷史清單；這一步不需要 Telegram 登入：

```sh
python bridge.py --config /private/config.json preview --output /private/history.json
```

同時產生 CSV，包含使用者 ID、案件、收錄原因和日誌連結。核對後，才批准該份清單的 SHA-256。

```sh
python bridge.py --config /private/config.json login
```

在終端機完成專用帳號登入。將輸出的帳號及 hexDrill 數字 ID 填回設定。若群組保留主持人的帳號，將其 ID 加入 `allowed_observers`；其他成員加入會令同步停止。

```sh
python bridge.py --config /private/config.json init
python bridge.py --config /private/config.json approve-history --file /private/history.json --sha256 REVIEWED_SHA256
```

請先讓 hexDrill 管理員確認：這個帳號只可在指定群組修改 `+spb`，而且該清單只由此程式寫入。否則程式無法分辨同一筆封禁是否也由別人手動加入。確認後才把 `receiver_scope_confirmed` 和 `live` 設為 `true`：

```sh
python bridge.py --config /private/config.json run
```

首次啟用前須在接收端驗證一組封禁及解除封禁回覆。程式按目前 hexDrill 的繁體中文回覆確認成功，格式改變會停止同步。

## 停止與復原

停止工作程序即可停止發送。`live: false` 關閉 `run` 模式；`watch` 模式的核准案件同步及 `/queue resume` 另行控制。保留 `state.db`，不要用刪除紀錄的方式解除錯誤。

每次只處理一條指令，收到指定機器人對該指令的成功回覆才記錄完成。斷線或回覆不明時停止；重啟後會尋找原訊息與回覆，不盲目重送。若仍無法確認，先檢查私密群組與 hexDrill 清單，再處理紀錄。沒有自動略過未確認指令的選項。

同一個人有多筆有效案件時只封禁一次；全部不再符合條件才解除，而且只解除本程式確認加入的項目。每小時最多 30 次、每日 150 次，間隔至少 20 秒；突然新增或移除超過 20 人會停止供檢查。歷史批次可經預覽批准，但仍受發送速率限制。

來源資料庫以唯讀方式開啟，讀取失敗會停止，不會當作名單清空。這不能防止同一 Toolforge 帳號被入侵後讀取 session 或竄改程式。接收端的權限限制須由 hexDrill 自身執行。

測試不連線 Telegram，也不發送訊息：

```sh
python -m unittest discover -s bridge -v
```

## SPB-WATCH 查詢與通知

在私密設定加入 `operator_id`（主持人的 Telegram 數字 ID），以 `watch` 啟動：

```sh
python bridge.py --config /private/config.json watch
```

這個模式接收指令、核對回覆及發送通知，也處理核准的 WATCH 案件。與 `run` 共用 session lock，不能同時啟動。

主持人可私訊 `/queue` 查看剩餘寫入，或用 `/queue resume` 重試最近一次明確失敗的操作。收到 hexDrill 對原指令的 `✅ +spb：已加入，封鎖（永久）` 回覆後，才繼續其餘已批准的歷史清單及 WATCH 案件。授權人員可在指定報告群使用這兩個指令。沒有失敗項目時，以第一筆待寫入項目試送。

每次重試都重新核對原失敗回覆、群組及來源案件。`⚠️ 操作失敗，請稍後再試。` 會暫停佇列，並通知主持人私訊及報告群；未收到回覆或發送結果不明時也暫停，但不允許盲目重送。再次輸入 `/queue resume` 才會重試明確失敗的操作。重啟會保留進度，重複指令不會多送一筆。未批准的歷史案件不會納入；此佇列沒有每小時、每日或固定間隔限制；每次只送一筆，確認成功後立即繼續。Telegram 要求冷卻時會等待。

主持人可私訊 `/sudo ID` 授權、`/sudo del ID` 撤權、`/sudo list` 查看名單。授權包含查詢、清除訊息、入群、WATCH 案件核准及恢復佇列；授權者不能再授權他人。啟用 `owner_only_private` 時，一般查詢及佇列指令須在指定報告群使用。

私訊或指定報告群使用 `/sync sync:追蹤碼`、`/sync #案件碼` 或 `/sync 使用者ID`，也支援 `/spb_sync`。啟動時略過既有聊天指令；重啟不重播已處理指令。轉寄、編輯及透過其他 bot 發出的指令不受理。

SPB 主機器人也支援主持人在私訊用 `/sync`，需設定 `HEXDRILL_STATE_DB` 指向同步帳本。主機器人唯讀查詢報表；超過 90 秒未更新會標示舊資料。封禁成功只以 hexDrill 的有效回覆為準。

未確認或失敗的操作會各通知主持人私訊及指定群組一次。群組成員無法核對時停止群組輸出，仍嘗試私訊。通知的 Telegram 發送結果不明時，只查找原訊息，不盲目重送；`watch_outbox` 保留兩個目的地的送達結果及錯誤。群組權限及來源讀取錯誤另外通知主持人。

### 私訊

私密設定的 `dm_enabled: true` 配合 SPB 的 `WATCH_DM_ENABLED=1` 啟用私訊檢測。只收啟用後的新訊息，略過 bot、平台帳號、轉寄和編輯訊息。SPB-WATCH 寫自己的收件匣；SPB 後臺讀取後，以當時的全域門檻判斷，不使用群組覆寫值。模型達門檻才自動加入 netban；0.5 以上未達門檻、或只命中 REGEX 的訊息交主持人審核。全域門檻低於 0.5 或模型讀取失敗時停止處理。

主持人、授權者及項目保護名單不自動封禁。每人每日最多檢測 20 條、總共 200 條；一天第 21 筆起的私訊案件一律送審。自動結果不訓練模型。審核及撤銷沿用 SPB 案件流程，但不對私訊呼叫群組封禁 API。

正常訊息每天只回覆一次角色帳號說明與討論群連結。私訊原文只留在私密收件匣，正常訊息原文一天後清除；公開案件只記錄來源和分數。只有主持人可私訊 `/evidence #案件碼` 查閱原文，授權查詢不包含此權限。

`/queue flush` 會固定當下已核准的剩餘名單，連續送出，之後核對每筆回覆。最多 200 筆；已送出的指令不會重送。收到失敗回覆或發送結果不明時停止剩餘發送，Telegram 限流則等待。此批執行期間新增的核准案件留待下一批。
