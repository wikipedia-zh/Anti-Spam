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

停止工作程序即可停止發送。改成 `live: false` 後重新啟動也不會發送。保留 `state.db`，不要用刪除紀錄的方式解除錯誤。

每次只處理一條指令，收到指定機器人對該指令的成功回覆才記錄完成。斷線或回覆不明時停止；重啟後會尋找原訊息與回覆，不盲目重送。若仍無法確認，先檢查私密群組與 hexDrill 清單，再處理紀錄。沒有自動略過未確認指令的選項。

同一個人有多筆有效案件時只封禁一次；全部不再符合條件才解除，而且只解除本程式確認加入的項目。每小時最多 30 次、每日 150 次，間隔至少 20 秒；突然新增或移除超過 20 人會停止供檢查。歷史批次可經預覽批准，但仍受發送速率限制。

來源資料庫以唯讀方式開啟，讀取失敗會停止，不會當作名單清空。這不能防止同一 Toolforge 帳號被入侵後讀取 session 或竄改程式。接收端的權限限制須由 hexDrill 自身執行。

測試不連線 Telegram，也不發送訊息：

```sh
python -m unittest discover -s bridge -v
```
