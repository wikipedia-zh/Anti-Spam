(() => {
  'use strict';
  const app = window.Telegram?.WebApp;
  const $ = id => document.getElementById(id);
  const copy = {
    'zh-Hant': {
      heading:'群組設定',protection:'群組防護',messages:'訊息管理',model:'垃圾訊息判斷',threshold:'封禁門檻',useGlobal:'跟隨全域設定',customThreshold:'本群門檻（0.50–0.99）',thresholdHint:'降低門檻會增加本群封禁，也可能增加誤判。聯防仍依全域標準收錄。',scope:'設定只套用到本群。儲存前會列出變更供你確認。',discard:'取消變更',review:'查看變更',save:'儲存設定',back:'返回',reload:'重新讀取',loading:'正在確認權限…',ready:'只有本群管理員可以修改設定。',saved:'已儲存。',saving:'儲存中…',on:'開啟',off:'關閉',global:'全域',custom:'本群',readOnly:'只有項目維護員可以修改門檻。本群門檻不會降低聯防標準。',pending:'有 {n} 項變更尚未儲存',leave:'放棄尚未儲存的變更？',open:'請在 Telegram 群組輸入 /settings，使用機器人提供的連結開啟。',session_expired:'連結或登入已過期。請回到群組重新輸入 /settings。',forbidden:'無法存取本群設定。請確認你和機器人仍是本群管理員。',settings_changed:'另一位管理員已修改設定。請重新讀取後再調整。',rate_limited:'操作太頻繁，請稍候一分鐘再試。',invalid_settings:'請檢查輸入的設定。',invalid_request:'無法讀取這項操作，請重新開啟設定。',temporarily_unavailable:'暫時無法連線，請稍後再試。',save_failed:'未能確認儲存結果。可重試同一次儲存，或重新讀取設定。',invalidThreshold:'請輸入 0.50 至 0.99 之間的數值。',unnamed:'群組',
      modules:{flood:['洗版防護','偵測短時間內大量發訊，並禁言發訊者。'],guestban:['訪客機器人','封禁並刪除非群組成員的機器人發言。'],captcha:['入群驗證','要求新成員完成驗證，逾時會移出群組。'],netban:['聯防黑名單','套用符合聯防條件的封禁紀錄。'],nohalal:['文字與名稱規則','啟用 NoHalal 現有的文字及名稱檢查。'],nocontact:['聯絡人卡片','攔截聯絡人卡片，並封禁發送者。'],novoice:['語音訊息','攔截語音訊息，並封禁發送者。'],noexec:['可執行檔案','攔截可執行檔案，並封禁發送者。'],nosm:['系統訊息','刪除入群、退群等 Telegram 系統訊息。'],cmdclean:['指令清理','刪除越權指令；24 小時內重犯會禁言 5 分鐘。']}
    },
    'zh-Hans': {
      heading:'群组设置',protection:'群组防护',messages:'消息管理',model:'垃圾消息判断',threshold:'封禁阈值',useGlobal:'跟随全局设置',customThreshold:'本群阈值（0.50–0.99）',thresholdHint:'降低阈值会增加本群封禁，也可能增加误判。联防仍按全局标准收录。',scope:'设置只应用到本群。保存前会列出变更供你确认。',discard:'取消变更',review:'查看变更',save:'保存设置',back:'返回',reload:'重新读取',loading:'正在确认权限…',ready:'只有本群管理员可以修改设置。',saved:'已保存。',saving:'保存中…',on:'开启',off:'关闭',global:'全局',custom:'本群',readOnly:'只有项目维护员可以修改阈值。本群阈值不会降低联防标准。',pending:'有 {n} 项变更尚未保存',leave:'放弃尚未保存的变更？',open:'请在 Telegram 群组输入 /settings，使用机器人提供的链接打开。',session_expired:'链接或登录已过期。请回到群组重新输入 /settings。',forbidden:'无法访问本群设置。请确认你和机器人仍是本群管理员。',settings_changed:'另一位管理员已修改设置。请重新读取后再调整。',rate_limited:'操作太频繁，请稍候一分钟再试。',invalid_settings:'请检查输入的设置。',invalid_request:'无法读取这项操作，请重新打开设置。',temporarily_unavailable:'暂时无法连接，请稍后再试。',save_failed:'未能确认保存结果。可重试同一次保存，或重新读取设置。',invalidThreshold:'请输入 0.50 至 0.99 之间的数值。',unnamed:'群组',
      modules:{flood:['刷屏防护','检测短时间内大量发消息，并禁言发送者。'],guestban:['访客机器人','封禁并删除非群组成员的机器人发言。'],captcha:['入群验证','要求新成员完成验证，超时会移出群组。'],netban:['联防黑名单','应用符合联防条件的封禁记录。'],nohalal:['文字与名称规则','启用 NoHalal 现有的文字及名称检查。'],nocontact:['联系人卡片','拦截联系人卡片，并封禁发送者。'],novoice:['语音消息','拦截语音消息，并封禁发送者。'],noexec:['可执行文件','拦截可执行文件，并封禁发送者。'],nosm:['系统消息','删除入群、退群等 Telegram 系统消息。'],cmdclean:['指令清理','删除越权指令；24 小时内重犯会禁言 5 分钟。']}
    },
    en: {
      heading:'Group settings',protection:'Group protection',messages:'Messages',model:'Spam detection',threshold:'Ban threshold',useGlobal:'Use global threshold',customThreshold:'Group threshold (0.50–0.99)',thresholdHint:'A lower threshold increases bans in this group and may increase mistakes. Shared bans still require the global criteria.',scope:'Changes apply to this group. Review them before saving.',discard:'Discard changes',review:'Review changes',save:'Save settings',back:'Back',reload:'Reload settings',loading:'Checking access…',ready:'Only this group’s admins can change these settings.',saved:'Saved.',saving:'Saving…',on:'On',off:'Off',global:'Global',custom:'Group',readOnly:'Only project maintainers can change this threshold. It does not lower the criteria for shared bans.',pending:'{n} unsaved changes',leave:'Discard your unsaved changes?',open:'Send /settings in your Telegram group, then open the link from the bot.',session_expired:'This link or session has expired. Send /settings in the group again.',forbidden:'Access unavailable. Check that you and the bot are still admins in this group.',settings_changed:'Another admin changed these settings. Reload before making changes.',rate_limited:'Too many requests. Please wait a minute and try again.',invalid_settings:'Please check the settings you entered.',invalid_request:'This request could not be read. Please reopen settings.',temporarily_unavailable:'Unable to connect. Please try again later.',save_failed:'The save could not be confirmed. Retry the same save or reload settings.',invalidThreshold:'Enter a number from 0.50 to 0.99.',unnamed:'Group',
      modules:{flood:['Flood protection','Mute users who send too many messages in a short time.'],guestban:['Guest bots','Delete posts and ban bots that are not group members.'],captcha:['Join verification','Ask new members to verify. Remove them if time runs out.'],netban:['Shared bans','Apply bans that meet the network’s eligibility rules.'],nohalal:['Text and name rules','Enable the existing NoHalal text and name checks.'],nocontact:['Contact cards','Block contact cards and ban the sender.'],novoice:['Voice messages','Block voice messages and ban the sender.'],noexec:['Executable files','Block executable files and ban the sender.'],nosm:['Service messages','Remove Telegram service messages, such as joins and departures.'],cmdclean:['Command cleanup','Delete unauthorized commands. Repeating within 24 hours triggers a 5-minute mute.']}
    }
  };
  Object.assign(copy['zh-Hant'], {customText:'群組自訂文字',otTemplate:'離題提醒 · /ot',otHint:'用 {user} 提及用戶，{count} 顯示目前警告數。支援原有的 Telegram HTML 格式。',useDefaultText:'使用預設文字',otButtons:'連結按鈕格式：',insertUser:'插入用戶',insertCount:'插入警告數',defaultText:'預設',invalid_template:'文字不可空白或超過 3500 字元；請檢查按鈕格式、網址及提及次數。'});
  Object.assign(copy['zh-Hans'], {customText:'群组自定义文字',otTemplate:'离题提醒 · /ot',otHint:'用 {user} 提及用户，{count} 显示目前警告数。支持原有的 Telegram HTML 格式。',useDefaultText:'使用默认文字',otButtons:'链接按钮格式：',insertUser:'插入用户',insertCount:'插入警告数',defaultText:'默认',invalid_template:'文字不可为空或超过 3500 字符；请检查按钮格式、网址及提及次数。'});
  Object.assign(copy.en, {customText:'Group messages',otTemplate:'Off-topic notice · /ot',otHint:'Use {user} to mention the user and {count} for their warning count. Existing Telegram HTML formatting is supported.',useDefaultText:'Use default text',otButtons:'Link button format:',insertUser:'Insert user',insertCount:'Insert count',defaultText:'Default',invalid_template:'Enter up to 3500 characters. Check button syntax, URLs and the number of user mentions.'});
  const language = navigator.language.toLowerCase();
  let lang = language.startsWith('zh') ? (/hans|cn|sg/.test(language) ? 'zh-Hans' : 'zh-Hant') : 'en';
  let token = '', original = null, draft = null, globalThreshold = 0, canEdit = false;
  let hostMode = false;
  let busy = false, blocked = false, pendingPatch = null, statusKey = 'loading', statusError = false;
  const t = key => copy[lang][key] || copy[lang].temporarily_unavailable;
  const groups = {protection:['flood','guestban','captcha','netban','nohalal'],messages:['nocontact','novoice','noexec','nosm','cmdclean']};
  const labels = {flood:'Flood',guestban:'GuestBan',captcha:'Captcha',netban:'Netban',nohalal:'NoHalal',nocontact:'NoContact',novoice:'NoVoice',noexec:'NoExec',nosm:'NoSM',cmdclean:'CmdClean'};
  function changes() {
    if (!original) return {};
    const result = {};
    for (const key of Object.keys(original.modules)) if (draft.modules[key] !== original.modules[key]) result[key] = draft.modules[key];
    if (canEdit && draft.threshold_override !== original.threshold_override) result.threshold_override = draft.threshold_override;
    if (Object.hasOwn(original,'ot_template') && draft.ot_template !== original.ot_template) result.ot_template = draft.ot_template;
    return result;
  }
  function status(key, error = false) {
    statusKey = key; statusError = error;
    $('status').textContent = t(key); $('status').classList.toggle('error', error);
  }
  function thresholdText(value) {
    if (!Number.isFinite(value)) return '—';
    const text = String(value);
    return (text.split('.')[1]?.length ?? 0) < 2 ? value.toFixed(2) : text;
  }
  function theme() {
    const dark = app?.initData ? app.colorScheme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    if (app?.initData && app.isVersionAtLeast?.('6.1')) {
      app.setHeaderColor(dark ? '#0c1319' : '#e9edef');
      app.setBackgroundColor(dark ? '#0c1319' : '#e9edef');
    }
  }
  function render() {
    if (hostMode) { window.SPBHost.setLanguage(lang); return; }
    document.documentElement.lang = lang; document.title = `${t('heading')} — SPB`;
    document.querySelectorAll('[data-text]').forEach(el => { el.textContent = t(el.dataset.text); });
    document.querySelectorAll('[data-lang]').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.lang === lang)));
    status(statusKey, statusError);
    if (!draft) return;
    $('group-name').textContent = original.title || `${t('unnamed')} ${original.chat_id}`;
    for (const [section, keys] of Object.entries(groups)) {
      $(section).replaceChildren();
      for (const key of keys) {
        const row = document.createElement('div'); row.className = 'setting-row';
        const content = document.createElement('div'); content.className = 'setting-copy';
        const heading = document.createElement('h3'); heading.id = `label-${key}`; heading.textContent = copy[lang].modules[key][0];
        const code = document.createElement('span'); code.className = 'module-key'; code.textContent = labels[key]; heading.append(code);
        const desc = document.createElement('p'); desc.id = `desc-${key}`; desc.textContent = copy[lang].modules[key][1];
        const button = document.createElement('button'); button.type = 'button'; button.className = 'switch'; button.dataset.module = key;
        button.setAttribute('role','switch'); button.setAttribute('aria-labelledby',heading.id); button.setAttribute('aria-describedby',desc.id);
        button.addEventListener('click', () => { draft.modules[key] = !draft.modules[key]; pendingPatch = null; update(); });
        content.append(heading,desc); row.append(content,button); $(section).append(row);
      }
    }
    $('threshold-controls').hidden = !canEdit;
    $('use-global').checked = draft.threshold_override === null;
    $('threshold-input').value = draft.threshold_override ?? globalThreshold;
    $('custom-text-section').hidden = !Object.hasOwn(original,'ot_template');
    $('ot-default').checked = draft.ot_template === null;
    $('ot-template').value = draft.ot_template ?? original.default_ot_template ?? '';
    update();
  }
  function update() {
    if (!draft) return;
    const count = Object.keys(changes()).length;
    document.querySelectorAll('[data-module]').forEach(el => {
      el.setAttribute('aria-checked',String(draft.modules[el.dataset.module])); el.disabled = busy || blocked;
    });
    $('threshold-note').textContent = canEdit ? (draft.threshold_override === null ? t('useGlobal') : t('customThreshold')) : t('readOnly');
    $('threshold-value').textContent = thresholdText(draft.threshold_override ?? globalThreshold);
    $('use-global').disabled = busy || blocked;
    $('threshold-input').disabled = busy || blocked || draft.threshold_override === null;
    $('ot-default').disabled = busy || blocked;
    $('ot-template').disabled = busy || blocked || draft.ot_template === null;
    $('ot-user').disabled = busy || blocked || draft.ot_template === null;
    $('ot-count-token').disabled = busy || blocked || draft.ot_template === null;
    $('ot-count').textContent = `${$('ot-template').value.length} / 3500`;
    $('save-bar').hidden = !count;
    $('unsaved').textContent = t('pending').replace('{n}',count);
    $('review').disabled = busy || blocked; $('discard').disabled = busy;
    $('save').disabled = busy || blocked; $('cancel').disabled = busy;
    $('reload').disabled = busy;
    $('save').textContent = t(busy ? 'saving' : 'save');
    if (app?.initData && app.isVersionAtLeast?.('6.2')) {
      if (count) app.enableClosingConfirmation(); else app.disableClosingConfirmation();
    }
    if (app?.initData && app.MainButton) {
      app.MainButton.setText(t('review'));
      if (count && !blocked && !busy && !$('review-dialog').open && !$('discard-dialog').open) app.MainButton.show(); else app.MainButton.hide();
    }
  }
  async function request(route, method = 'GET', body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 28000);
    try {
      const response = await fetch(`api.php?route=${route}`,{method,headers:{'Content-Type':'application/json',...(token ? {Authorization:`Bearer ${token}`} : {})},credentials:'omit',cache:'no-store',redirect:'error',signal:controller.signal,...(body ? {body:JSON.stringify(body)} : {})});
      let data;
      try { data = await response.json(); } catch { throw new Error(method === 'PATCH' ? 'save_failed' : 'temporarily_unavailable'); }
      if (!response.ok) throw new Error(Object.hasOwn(copy.en,data.error) ? data.error : 'temporarily_unavailable');
      return data;
    } catch (error) {
      if (Object.hasOwn(copy.en,error.message)) throw error;
      throw new Error(method === 'PATCH' ? 'save_failed' : 'temporarily_unavailable');
    } finally { clearTimeout(timer); }
  }
  function fail(error, saving = false) {
    if (hostMode) { window.SPBHost.fail(error); return; }
    const key = error.message;
    blocked = ['session_expired','forbidden','settings_changed'].includes(key);
    if (key === 'session_expired' || key === 'forbidden') token = '';
    status(key,true); $('reload').hidden = !token;
    if (saving) $('save-error').textContent = t(key);
    update();
  }
  async function load() {
    busy = true; update();
    try {
      const data = await request('settings');
      original = data.settings; draft = structuredClone(original);
      globalThreshold = data.global_threshold; canEdit = data.can_edit_threshold;
      blocked = false; pendingPatch = null;
      $('settings').hidden = false; $('reload').hidden = true;
      status('ready'); render();
    } catch (error) { fail(error); }
    finally { busy = false; update(); }
  }
  function valueLabel(key, value) {
    if (key === 'ot_template') return value === null ? `${t('defaultText')}\n${original.default_ot_template}` : value;
    if (key !== 'threshold_override') return t(value ? 'on' : 'off');
    return value === null ? `${t('global')} · ${thresholdText(globalThreshold)}` : `${t('custom')} · ${thresholdText(value)}`;
  }
  function review() {
    if (busy || blocked || !Object.keys(changes()).length) return;
    if (canEdit && draft.threshold_override !== null && (!$('threshold-input').checkValidity() || !Number.isFinite(draft.threshold_override))) {
      status('invalidThreshold',true); $('threshold-input').focus(); return;
    }
    if (Object.hasOwn(changes(),'ot_template') && draft.ot_template !== null && (!draft.ot_template.trim() || !$('ot-template').checkValidity())) {
      status('invalid_template',true); $('ot-template').focus(); return;
    }
    $('review-group').textContent = $('group-name').textContent;
    $('changes').replaceChildren(); $('save-error').textContent = '';
    for (const [key,value] of Object.entries(changes())) {
      const term = document.createElement('dt'); term.textContent = key === 'ot_template' ? t('otTemplate') : key === 'threshold_override' ? t('threshold') : copy[lang].modules[key][0];
      const detail = document.createElement('dd');
      const before = valueLabel(key,['threshold_override','ot_template'].includes(key) ? original[key] : original.modules[key]);
      detail.textContent = key === 'ot_template' ? `${before}\n↓\n${valueLabel(key,value)}` : `${before} → ${valueLabel(key,value)}`;
      if (key === 'ot_template') detail.className = 'text-change';
      $('changes').append(term,detail);
    }
    $('review-dialog').showModal(); update();
  }
  function confirmDiscard() {
    if (busy || $('discard-dialog').open) return Promise.resolve(false);
    return new Promise(resolve => {
      const dialog = $('discard-dialog'); dialog.returnValue = '';
      dialog.addEventListener('close', () => { update(); resolve(dialog.returnValue === 'discard'); },{once:true});
      dialog.showModal(); update();
    });
  }
  async function discard() {
    if (!(await confirmDiscard())) return;
    draft = structuredClone(original); pendingPatch = null; render();
  }
  async function save() {
    if (busy || blocked) return;
    pendingPatch ??= {request_id:crypto.randomUUID(),expected_revision:original.revision,changes:changes()};
    busy = true; $('save-error').textContent = ''; update();
    try {
      original = await request('settings','PATCH',pendingPatch);
      draft = structuredClone(original); pendingPatch = null;
      $('review-dialog').close(); status('saved'); $('reload').hidden = true; render();
    } catch (error) { fail(error,true); }
    finally { busy = false; update(); }
  }
  document.querySelectorAll('[data-lang]').forEach(el => el.addEventListener('click', () => { lang = el.dataset.lang; render(); }));
  $('use-global').addEventListener('change', () => {
    draft.threshold_override = $('use-global').checked ? null : globalThreshold;
    $('threshold-input').value = draft.threshold_override ?? globalThreshold; pendingPatch = null; update();
  });
  $('threshold-input').addEventListener('input', () => { draft.threshold_override = $('threshold-input').valueAsNumber; pendingPatch = null; update(); });
  $('ot-default').addEventListener('change', () => {
    draft.ot_template = $('ot-default').checked ? null : $('ot-template').value;
    $('ot-template').value = draft.ot_template ?? original.default_ot_template;
    pendingPatch = null; update();
  });
  $('ot-template').addEventListener('input', () => { draft.ot_template = $('ot-template').value; pendingPatch = null; update(); });
  for (const [id,marker] of [['ot-user','{user}'],['ot-count-token','{count}']]) {
    $(id).addEventListener('click', () => {
      const input = $('ot-template');
      if (input.value.length - (input.selectionEnd - input.selectionStart) + marker.length > input.maxLength) return;
      input.setRangeText(marker,input.selectionStart,input.selectionEnd,'end');
      draft.ot_template = input.value; pendingPatch = null; update(); input.focus();
    });
  }
  $('review').addEventListener('click',review); $('discard').addEventListener('click',discard);
  $('save').addEventListener('click',save); $('cancel').addEventListener('click', () => $('review-dialog').close());
  $('review-dialog').addEventListener('cancel', event => { if (busy) event.preventDefault(); });
  $('review-dialog').addEventListener('close',update);
  $('keep-editing').addEventListener('click', () => $('discard-dialog').close());
  $('confirm-discard').addEventListener('click', () => $('discard-dialog').close('discard'));
  $('reload').addEventListener('click', async () => { if (!Object.keys(changes()).length || await confirmDiscard()) load(); });
  document.querySelector('.wordmark').addEventListener('click', async event => {
    if (busy) { event.preventDefault(); return; }
    if (!Object.keys(changes()).length) return;
    event.preventDefault(); const href = event.currentTarget.href;
    if (await confirmDiscard()) { draft = structuredClone(original); update(); location.assign(href); }
  });
  window.addEventListener('beforeunload', event => { if (Object.keys(changes()).length) { event.preventDefault(); event.returnValue = ''; } });
  async function back() {
    if (busy) return;
    if ($('discard-dialog').open) { $('discard-dialog').close(); return; }
    if ($('review-dialog').open) { $('review-dialog').close(); return; }
    if (!Object.keys(changes()).length || await confirmDiscard()) {
      draft = original ? structuredClone(original) : null; update(); app?.close();
    }
  }
  async function start() {
    theme(); render();
    if (!app?.initData) { status('open'); return; }
    app.ready(); app.expand(); app.onEvent('themeChanged',theme);
    if (app.isVersionAtLeast?.('6.1')) { app.BackButton.show(); app.BackButton.onClick(back); }
    app.MainButton?.onClick(review);
    try {
      const session = await request('session','POST',{init_data:app.initData});
      token = session.token;
      setTimeout(() => { token = ''; blocked = true; fail(new Error('session_expired')); },Math.max(0,session.expires_at * 1000 - Date.now()));
      if (session.scope === 'host') {
        hostMode = true;
        await window.SPBHost.start(request,lang,() => { token = ''; });
        return;
      }
      await load();
    } catch (error) { fail(error); }
  }
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',theme);
  start();
})();
