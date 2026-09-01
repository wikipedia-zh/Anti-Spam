---
layout: default
title: "Spam Protection Bot — 使用規範 / Terms of Use"
description: "Spam Protection Bot（SPB）使用規範"
---

# Spam Protection Bot — 使用規範 / Terms of Use

**生效日期 / Effective date:** 2026-08-15
**最後更新 / Last updated:** 2026-08-28

本文件同時提供繁體中文與英文版本。若兩者有歧義，以繁體中文版為準。
This document is provided in Traditional Chinese and English. In case of ambiguity, the Traditional Chinese version prevails.

尋找指令說明與各模組介紹？請見[使用指南](./guide.html)（繁體中文／简体中文）。
Looking for a command and module reference instead? See the [User Guide](./guide.html) (Traditional/Simplified Chinese).

---

## 繁體中文

### 1. 服務說明

Spam Protection Bot（下稱「SPB」或「本機器人」）是一個由志願者營運的 Telegram 反垃圾訊息機器人，免費提供給群組使用。本項目並非 Telegram 官方服務，亦不隸屬於 Telegram。

本服務按「現狀」提供，不保證可用性、正確性或持續營運。

### 2. 適用範圍

當您將本機器人加入群組，或在群組中使用本機器人的任何功能時，即表示群組管理團隊與使用者同意本規範。

本規範適用於：
- 邀請本機器人進入群組的帳號；
- 該群組的擁有者與管理團隊；
- 在群組中使用本機器人指令的所有使用者。

### 3. 使用條件 {#eligibility}

- **本服務僅供公開群組使用。** 私密（非公開）群組並非垃圾訊息散布的對象，並無使用本服務之必要；為合理配置有限資源，本項目不對私密群組提供服務，並保留隨時終止對此類群組服務的權利。
- 本機器人需要「刪除訊息」與「封禁使用者」權限才能運作。若權限不足，部分或全部功能將無法使用。
- 群組管理員可自行決定啟用或停用各項模組，並承擔該設定造成的後果。
- 本機器人的判斷以自動化模型與規則為基礎，可能產生誤判。群組應自行保留人工覆核的能力。

### 4. 禁止行為 {#prohibited}

不得從事下列行為：

1. 濫用本機器人的指令，包括但不限於：對非垃圾內容大量使用檢舉或封禁指令、以指令騷擾其他使用者；
2. 意圖污染訓練資料，包括提交虛假的檢舉、將正常訊息標記為垃圾訊息，或以其他方式使模型產生偏差；
3. 規避、干擾或試圖破壞本機器人的封禁、審核或紀錄機制；
4. 將本機器人用於與反垃圾訊息無關的用途，例如作為一般群組管理工具對特定使用者進行針對性處置；
5. 利用本機器人的跨群組封禁同步（Netban）機制，對其他群組施加不當影響；
6. 對本項目的維護人員、審核員或基礎設施進行攻擊、騷擾或未經授權的存取。

### 5. 內容處理與資料

- 為進行垃圾訊息判斷，本機器人會讀取群組中的訊息內容。
- 被判定為垃圾訊息、或經人工審核批准的訊息內容，可能被保存並用於改進判斷模型。
- 封禁與管理操作會記錄於公開的日誌頻道，內容包含使用者 ID、操作原因與相關訊息片段。
- 請勿在啟用本機器人的群組中傳送敏感個人資料。

### 6. 封禁與跨群組同步

- 群組管理員的封禁僅在該群組生效。
- 達到本項目全域門檻的自動判定、以及經項目組審核批准的檢舉，會被列入本項目的共用封禁名單。
- 已啟用 Netban 模組的群組，會接收並執行共用封禁名單中的封禁。是否接收由各群組自行決定。

### 7. 申訴

對封禁有異議者，請透過 [@SEELE_01_BOT](https://t.me/SEELE_01_BOT) 提出申訴。此為唯一受理管道，透過其他方式提出的申訴不予處理。

### 8. 服務終止與拒絕服務 {#termination}

本服務由專案擁有者自願提供，並無提供義務。**專案擁有者保留隨時、依其單方且完全的裁量權（sole and absolute discretion），拒絕或終止對任何使用者或群組提供服務的權利，無論是否給予理由。**

在下列情況下（但不限於此），可能終止或拒絕服務：

- 違反本規範；
- 基於信任與安全（trust and safety）之考量——包括但不限於濫用、規避封禁、危害其他使用者或公眾、或任何專案擁有者認為構成風險的行為；
- 經評估認為繼續提供服務將損害本項目、其他使用者或公眾利益；
- 基於資源管理或服務完整性之考量。

**基於信任與安全所作的封禁或拒絕服務，專案擁有者沒有義務提供公開說明或理由，亦沒有義務事先通知。** 相關內部紀錄將被保留。專案擁有者保留在必要時公開更多資訊的權利，但無提供之義務。此類決定為最終決定；唯一的申訴管道為 [@SEELE_01_BOT](https://t.me/SEELE_01_BOT)，且是否受理由專案擁有者裁量。

### 9. 免責聲明

本項目對因使用或無法使用本服務所導致的任何損失不承擔責任，包括但不限於誤判封禁、漏判垃圾訊息、服務中斷或資料遺失。

### 10. 條款變更

本規範可能隨時修訂，修訂後的版本自公布時起生效。重大變更會於項目頻道公告。

---

## English

### 1. About the service

Spam Protection Bot ("SPB" or "the bot") is a volunteer-operated Telegram anti-spam bot, provided free of charge to groups. This project is not an official Telegram service and is not affiliated with Telegram.

The service is provided "as is", with no guarantee of availability, accuracy, or continued operation.

### 2. Scope

By adding the bot to a group, or by using any of its features in a group, the group's administrators and users agree to these terms.

These terms apply to:
- the account that invites the bot into a group;
- the owner and administration team of that group;
- all users who invoke the bot's commands in that group.

### 3. Conditions of use

- **The service is for public groups only.** Private (non-public) groups are not targets for spam distribution and have no need for the service; to allocate limited resources responsibly, the project does not serve private groups and reserves the right to terminate service to any such group at any time.
- The bot requires "delete messages" and "ban users" permissions to function. Without them, some or all features will not work.
- Group administrators decide which modules to enable, and are responsible for the consequences of those settings.
- The bot's decisions are based on automated models and rules and may be wrong. Groups should retain their own capacity for human review.

### 4. Prohibited conduct

You must not:

1. Abuse the bot's commands — including mass-reporting or mass-banning non-spam content, or using commands to harass other users;
2. Attempt to poison the training data, including submitting false reports, labelling legitimate messages as spam, or otherwise deliberately biasing the model;
3. Circumvent, interfere with, or attempt to compromise the bot's ban, review, or logging mechanisms;
4. Use the bot for purposes unrelated to anti-spam work, such as employing it as a general moderation tool to target specific users;
5. Exploit the cross-group ban synchronisation (Netban) mechanism to exert improper influence over other groups;
6. Attack, harass, or attempt unauthorised access to the project's maintainers, reviewers, or infrastructure.

### 5. Content handling and data

- The bot reads message content in the group in order to assess it for spam.
- Content judged to be spam, or approved through human review, may be retained and used to improve the classification model.
- Bans and moderation actions are recorded in a public log channel, including user IDs, the reason for the action, and relevant message excerpts.
- Do not send sensitive personal data in groups where the bot is active.

### 6. Bans and cross-group synchronisation

- A ban issued by a group administrator applies only to that group.
- Automated determinations that meet the project-wide threshold, and reports approved by the project team, are added to the project's shared ban list.
- Groups that have enabled the Netban module receive and enforce bans from that shared list. Whether to receive them is each group's own choice.

### 7. Appeals

If you disagree with a ban, appeal through [@SEELE_01_BOT](https://t.me/SEELE_01_BOT). This is the only channel through which appeals are accepted; appeals submitted by other means will not be processed.

### 8. Termination and refusal of service

This service is provided voluntarily by the project owner, who is under no obligation to provide it. **The project owner reserves the right, at any time and at their sole and absolute discretion, to refuse or terminate service to any user or group, with or without reason.**

Service may be terminated or refused where (without limitation):

- these terms have been breached;
- for trust-and-safety reasons — including but not limited to abuse, ban evasion, harm to other users or the public, or any conduct the owner considers a risk;
- continuing to provide service is judged harmful to the project, other users, or the public interest;
- termination is warranted by resource management or service integrity considerations.

**For bans or refusals of service made on trust-and-safety grounds, the project owner is under no obligation to provide any public explanation or reason, and no obligation to give prior notice.** Internal records are retained. The owner reserves the right, but is not obliged, to disclose further information. Such decisions are final; the only appeal channel is [@SEELE_01_BOT](https://t.me/SEELE_01_BOT), and whether an appeal is considered is at the owner's discretion.

### 9. Disclaimer

The project accepts no liability for any loss arising from use of, or inability to use, this service — including wrongful bans, spam that was not caught, service interruption, or data loss.

### 10. Changes to these terms

These terms may be revised at any time. Revised versions take effect when published. Significant changes will be announced in the project channel.

---

*項目交流群 / Project chat: [@SpamProtectionChat](https://t.me/SpamProtectionChat)*
*日誌頻道 / Log channel: [@SpamProtectionLogging](https://t.me/SpamProtectionLogging)*
