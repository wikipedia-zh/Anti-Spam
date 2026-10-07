(() => {
  'use strict';
  const words = {
    'zh-Hant': {
      title:'項目管理',overview:'總覽',cases:'案件',groups:'群組',people:'人員',audit:'操作記錄',queue:'待處理',
      search:'用戶 ID、群組 ID 或案件 ID',groupSearch:'群組名稱或 ID',find:'搜尋',refresh:'重新讀取',logout:'登出',
      loading:'正在讀取…',empty:'沒有符合條件的記錄。',previous:'上一頁',next:'下一頁',page:'頁',updated:'更新於',
      pending_reports:'待審舉報',pending_work:'待處理工作',failed_work:'曾失敗、待重試',pending_network:'待送聯防',
      known_groups:'記錄中的群組',rules:'封禁規則',global_threshold:'全域模型門檻',schema:'資料庫版本',version:'程式版本',
      note:'顯示目前保存的記錄。群組最近出現時間不代表機器人現在仍有管理權限。',
      modelNote:'模型在本群封禁與加入聯防，使用不同的判斷條件。',unknown:'未知',yes:'是',no:'否',global:'跟隨全域',
      id:'案件／操作 ID',action:'操作',chat_id:'群組 ID',target_user_id:'用戶 ID',status:'處理狀態',model_score:'模型分數',reason:'觸發原因',
      netban_eligible:'曾符合聯防條件',network_done:'已完成聯防工作',network_pending:'待送聯防工作',created_at:'記錄時間',evidence:'證據摘要',
      last_seen:'最近出現',netban:'接收聯防',spam_threshold_override:'本群門檻',service_denied:'已終止服務',
      user_id:'用戶 ID',role:'角色',added_by:'授權人',host:'項目主持人',maintainer:'維護員',reviewer:'審核員',
      actor_user_id:'操作人',source:'來源',detail:'變更摘要',reverted:'已撤銷',settings:'群組設定',command:'指令',
      kind:'工作種類',case_id:'案件 ID',attempts:'嘗試次數',next_attempt_at:'下次嘗試',last_error:'上次錯誤',
      session_expired:'登入已過期。請私訊機器人重新輸入 /manage。',forbidden:'只有項目主持人可以使用此面板。',
      temporarily_unavailable:'暫時無法讀取，請稍後再試。',rate_limited:'操作太頻繁，請稍候一分鐘再試。',
      signedOut:'已登出。請私訊機器人輸入 /manage 重新開啟。',cooldown:'Telegram 暫停接收操作，重試時間：'
    },
    en: {
      title:'Project management',overview:'Overview',cases:'Cases',groups:'Groups',people:'People',audit:'Activity',queue:'Pending work',
      search:'User, group or case ID',groupSearch:'Group name or ID',find:'Search',refresh:'Refresh',logout:'Sign out',
      loading:'Loading…',empty:'No matching records.',previous:'Previous',next:'Next',page:'Page',updated:'Updated',
      pending_reports:'Reports awaiting review',pending_work:'Pending work',failed_work:'Failed, awaiting retry',pending_network:'Pending shared bans',
      known_groups:'Recorded groups',rules:'Ban rules',global_threshold:'Global model threshold',schema:'Database version',version:'Code version',
      note:'These are saved records. A group’s last activity does not confirm the bot still has admin access.',
      modelNote:'Local model bans and shared bans have separate eligibility criteria.',unknown:'Unknown',yes:'Yes',no:'No',global:'Use global',
      id:'Case / action ID',action:'Action',chat_id:'Group ID',target_user_id:'User ID',status:'Status',model_score:'Model score',reason:'Reason',
      netban_eligible:'Qualified for shared bans',network_done:'Completed shared-ban jobs',network_pending:'Pending shared-ban jobs',created_at:'Recorded',evidence:'Evidence excerpt',
      last_seen:'Last seen',netban:'Receives shared bans',spam_threshold_override:'Group threshold',service_denied:'Service denied',
      user_id:'User ID',role:'Role',added_by:'Granted by',host:'Project host',maintainer:'Maintainer',reviewer:'Reviewer',
      actor_user_id:'Actor',source:'Source',detail:'Change summary',reverted:'Reverted',settings:'Group settings',command:'Command',
      kind:'Work type',case_id:'Case ID',attempts:'Attempts',next_attempt_at:'Next attempt',last_error:'Last error',
      session_expired:'Session expired. Send /manage to the bot in a private chat.',forbidden:'This panel is only available to the project host.',
      temporarily_unavailable:'Unable to load. Please try again later.',rate_limited:'Too many requests. Please wait a minute.',
      signedOut:'Signed out. Send /manage to the bot to open a new session.',cooldown:'Telegram cooldown ends at '
    }
  };
  words['zh-Hans'] = {...words['zh-Hant'],title:'项目管理',overview:'总览',cases:'案件',groups:'群组',people:'人员',audit:'操作记录',queue:'待处理',search:'用户 ID、群组 ID 或案件 ID',groupSearch:'群组名称或 ID',find:'搜索',refresh:'重新读取',logout:'退出登录',loading:'正在读取…',empty:'没有符合条件的记录。',previous:'上一页',next:'下一页',page:'页',updated:'更新于',pending_reports:'待审核举报',pending_work:'待处理工作',failed_work:'曾失败、等待重试',pending_network:'待发送联防',known_groups:'记录中的群组',rules:'封禁规则',global_threshold:'全局模型阈值',schema:'数据库版本',version:'程序版本',note:'显示目前保存的记录。群组最近出现时间不代表机器人现在仍有管理权限。',modelNote:'模型在本群封禁与加入联防，使用不同的判断条件。',unknown:'未知',yes:'是',no:'否',global:'跟随全局',chat_id:'群组 ID',target_user_id:'用户 ID',status:'处理状态',model_score:'模型分数',reason:'触发原因',netban_eligible:'曾符合联防条件',network_done:'已完成联防工作',network_pending:'待发送联防工作',created_at:'记录时间',evidence:'证据摘要',last_seen:'最近出现',netban:'接收联防',spam_threshold_override:'本群阈值',service_denied:'已终止服务',user_id:'用户 ID',role:'角色',added_by:'授权人',host:'项目主持人',maintainer:'维护员',reviewer:'审核员',actor_user_id:'操作人',source:'来源',detail:'变更摘要',reverted:'已撤销',settings:'群组设置',command:'指令',kind:'工作种类',case_id:'案件 ID',attempts:'尝试次数',next_attempt_at:'下次尝试',last_error:'上次错误',session_expired:'登录已过期。请私聊机器人重新输入 /manage。',forbidden:'只有项目主持人可以使用此面板。',temporarily_unavailable:'暂时无法读取，请稍后重试。',rate_limited:'操作太频繁，请稍候一分钟再试。',signedOut:'已退出登录。请私聊机器人输入 /manage 重新打开。',cooldown:'Telegram 暂停接收操作，重试时间：'};
  const views = ['overview','cases','groups','people','audit','queue'];
  const fields = {
    cases:['id','chat_id','target_user_id','action','status','model_score','reason','netban_eligible','network_done','network_pending','created_at'],
    groups:['chat_id','last_seen','netban','spam_threshold_override','service_denied'],
    people:['user_id','role','added_by','created_at'],audit:['id','source','actor_user_id','chat_id','action','reverted','created_at'],
    queue:['kind','case_id','chat_id','attempts','next_attempt_at']
  };
  const states={
    'zh-Hant':{auto_ban:'自動封禁',spam_ban:'人工封禁',pending_report:'待審舉報',report_approved:'舉報已批准',report_rejected:'舉報已拒絕',guest_bot_ban:'訪客機器人封禁',guest_invoker_ban:'訪客召喚者封禁',mute:'禁言',kick:'踢出群組',flood_mute:'洗版禁言',project_ban:'項目封禁',pending_review:'待審核',ban_done:'已封禁',done:'已完成',ban_pending:'待封禁',ban_failed:'封禁失敗',reversed:'已撤銷',reversal_pending:'正在撤銷',action_failed:'操作失敗',action_unconfirmed:'結果未確認',action_cancelled:'已取消',action_pending:'待執行'},
    'zh-Hans':{auto_ban:'自动封禁',spam_ban:'人工封禁',pending_report:'待审核举报',report_approved:'举报已批准',report_rejected:'举报已拒绝',guest_bot_ban:'访客机器人封禁',guest_invoker_ban:'访客召唤者封禁',mute:'禁言',kick:'踢出群组',flood_mute:'刷屏禁言',project_ban:'项目封禁',pending_review:'待审核',ban_done:'已封禁',done:'已完成',ban_pending:'待封禁',ban_failed:'封禁失败',reversed:'已撤销',reversal_pending:'正在撤销',action_failed:'操作失败',action_unconfirmed:'结果未确认',action_cancelled:'已取消',action_pending:'待执行'},
    en:{auto_ban:'Automatic ban',spam_ban:'Manual ban',pending_report:'Report awaiting review',report_approved:'Approved report',report_rejected:'Rejected report',guest_bot_ban:'Guest bot ban',guest_invoker_ban:'Guest invoker ban',mute:'Mute',kick:'Kick',flood_mute:'Flood mute',project_ban:'Project ban',pending_review:'Awaiting review',ban_done:'Banned',done:'Completed',ban_pending:'Ban pending',ban_failed:'Ban failed',reversed:'Reversed',reversal_pending:'Reversal pending',action_failed:'Action failed',action_unconfirmed:'Result unconfirmed',action_cancelled:'Cancelled',action_pending:'Action pending'}
  };
  Object.assign(words['zh-Hant'],{details:'查看記錄',manageRoles:'管理權限',editRoles:'修改權限',userSearch:'用戶 ID',queueSearch:'群組 ID 或案件 ID',auditSearch:'操作人 ID、群組 ID 或操作 ID'});
  Object.assign(words['zh-Hans'],{details:'查看记录',manageRoles:'管理权限',editRoles:'修改权限',userSearch:'用户 ID',queueSearch:'群组 ID 或案件 ID',auditSearch:'操作人 ID、群组 ID 或操作 ID'});
  Object.assign(words.en,{details:'View record',manageRoles:'Manage roles',editRoles:'Edit roles',userSearch:'User ID',queueSearch:'Group or case ID',auditSearch:'Actor, group or action ID'});
  for(const language of Object.keys(states)){
    const s=states[language];
    Object.assign(s,{auto_banned:s.ban_done,guest_bot_banned:s.ban_done,guest_invoker_banned:s.ban_done,approved_and_banned:s.ban_done,force_approved:s.ban_done});
    s.banned_delete_failed=language==='en'?'Banned; message deletion failed':language==='zh-Hans'?'已封禁，删消息失败':'已封禁，刪訊息失敗';
  }
  Object.assign(words['zh-Hant'],{groupSettings:'設定與自訂文字',openSettings:'前往群組設定 ↗',groupLinkNote:'此入口 5 分鐘內有效，只限你本人使用。開啟及儲存時會重新確認你和機器人的群組管理權限。',group_access_denied:'無法開啟此群設定。請確認你和機器人仍是該群管理員，且該群未被終止服務。'});
  Object.assign(words['zh-Hans'],{groupSettings:'设置与自定义文字',openSettings:'前往群组设置 ↗',groupLinkNote:'此入口 5 分钟内有效，仅限你本人使用。打开及保存时会重新确认你和机器人的群组管理权限。',group_access_denied:'无法打开此群设置。请确认你和机器人仍是该群管理员，且该群未被终止服务。'});
  Object.assign(words.en,{groupSettings:'Settings and group messages',openSettings:'Open group settings ↗',groupLinkNote:'This link is valid for 5 minutes and only works for you. Your group admin rights and the bot’s are checked again when opening and saving.',group_access_denied:'Unable to open this group’s settings. You and the bot must still be group admins, and the group must not be denied service.'});
  Object.assign(words['zh-Hant'],{caseDetails:'案件詳情與撤銷'});Object.assign(words['zh-Hans'],{caseDetails:'案件详情与撤销'});Object.assign(words.en,{caseDetails:'Case details and reversal'});
  let lang='zh-Hant',request,clearToken,root,view='overview',search='',filter='',offset=0,data=null,busy=false,closed=false,error='',groupLink=null;
  const t = key => words[lang][key] ?? key;
  const element = (tag,text,className) => {const el=document.createElement(tag); if(text!==undefined)el.textContent=text; if(className)el.className=className; return el;};
  function button(text,action) {const el=element('button',text);el.type='button';el.disabled=busy||closed;el.addEventListener('click',action);return el;}
  function date(value) {return value ? new Date(typeof value==='number'?value*1000:value).toLocaleString(lang) : t('unknown');}
  function value(key,v) {
    if(v===null||v===undefined)return key==='spam_threshold_override'?t('global'):t('unknown');
    if(['netban_eligible','netban','service_denied','reverted'].includes(key))return t(v?'yes':'no');
    if(['created_at','last_seen','next_attempt_at'].includes(key))return date(v);
    if(['role','source'].includes(key))return t(v);
    if(['action','status'].includes(key))return states[lang][v]||t(v);
    return String(v);
  }
  function render() {
    if(!root)return;
    document.documentElement.lang=lang;document.title=`${t('title')} — SPB`;
    document.querySelectorAll('[data-lang]').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.lang===lang)));
    root.replaceChildren(element('p','SPAM PROTECTION BOT / MANAGEMENT','eyebrow'));
    const heading=element('div',undefined,'host-heading');heading.append(element('h1',t('title')),button(t('logout'),logout));root.append(heading);
    const status=element('p',error?t(error):busy?t('loading'):data?`${t('updated')} ${date(data.updated_at)}`:'','host-meta');status.setAttribute('role','status');status.setAttribute('aria-live','polite');if(error)status.classList.add('host-error');root.append(status);
    if(closed)return;
    const nav=element('nav',undefined,'host-nav');nav.setAttribute('aria-label',t('title'));
    for(const name of views){const item=button(t(name),()=>{view=name;search='';filter='';offset=0;load();});if(name===view)item.setAttribute('aria-current','page');nav.append(item);}root.append(nav);
    if(filter)root.append(element('p',t({pending_review:'pending_reports',failed:'failed_work',network:'pending_network'}[filter]),'host-note'));
    const form=element('form',undefined,'host-tools');form.addEventListener('submit',event=>{event.preventDefault();if(!busy){search=input.value.trim();offset=0;load();}});
    const input=element('input');input.type='search';input.maxLength=100;input.value=search;input.disabled=busy;input.id='host-search';
    if(view!=='overview'){const label=element('label',t({groups:'groupSearch',people:'userSearch',queue:'queueSearch',audit:'auditSearch'}[view]||'search'));label.htmlFor=input.id;label.append(input);const submit=element('button',t('find'));submit.type='submit';submit.disabled=busy;form.append(label,submit);}
    form.append(button(t('refresh'),load));root.append(form);
    if(view==='people')root.append(button(t('manageRoles'),()=>editRole()));
    if(!data)return;
    if(view==='overview'){
      const summary=data.items[0];const grid=element('div',undefined,'host-summary');
      for(const key of ['pending_reports','pending_work','failed_work','pending_network']){const card=button('',()=>{view=key==='pending_reports'?'cases':'queue';filter={pending_reports:'pending_review',failed_work:'failed',pending_network:'network'}[key]||'';search='';offset=0;load();});card.className='host-stat';card.append(element('span',t(key)),element('strong',value(key,summary[key])));grid.append(card);}root.append(grid);
      const info=element('article',undefined,'host-record');const dl=element('dl');for(const key of ['known_groups','rules','global_threshold','version','schema'])dl.append(element('dt',t(key)),element('dd',value(key,summary[key])));info.append(dl);root.append(info);
      if(summary.telegram_not_before>Date.now()/1000)root.append(element('p',t('cooldown')+date(summary.telegram_not_before),'host-note'));
      root.append(element('p',t('modelNote'),'host-note'));
    }else{
      if(!data.items.length)root.append(element('p',t('empty')));
      for(const item of data.items){const card=element('article',undefined,'host-record');card.append(element('h3',item.target_name||item.title||t(item.role)||item.case_id||item.id||t(view)));
        const brief=[item.target_user_id??item.user_id??item.chat_id,item.status?value('status',item.status):item.kind||'',item.created_at?date(item.created_at):''].filter(v=>v!==''&&v!==undefined).join(' · ');card.append(element('p',brief,'host-meta'));
        const detail=element('details');detail.append(element('summary',t('details')));const dl=element('dl');for(const key of fields[view])dl.append(element('dt',t(key)),element('dd',value(key,item[key])));detail.append(dl);
        for(const key of ['evidence','detail','last_error'])if(item[key]){detail.append(element('h4',t(key)),element('pre',item[key]));}card.append(detail);if(view==='people'&&item.role!=='host')card.append(button(t('editRoles'),()=>editRole(item.user_id)));
        if(view==='cases')card.append(button(t('caseDetails'),()=>window.SPBCasePanel.open({request,language:lang,caseId:item.id,status:s=>value('status',s),onAuthError:fail,onSaved:load})));
        if(view==='groups'){
          card.append(button(t('groupSettings'),()=>prepareGroup(item.chat_id)));
          if(groupLink?.chat_id===item.chat_id){
            const link=element('a',t('openSettings'),'host-group-link');link.href=groupLink.url;
            link.addEventListener('click',event=>{if(window.Telegram?.WebApp?.openTelegramLink){event.preventDefault();window.Telegram.WebApp.openTelegramLink(link.href);}});
            card.append(link,element('p',t('groupLinkNote'),'host-note'));
          }
        }
        root.append(card);}
      const pager=element('div',undefined,'host-pagination');const previous=button(t('previous'),()=>{offset-=25;load();});previous.disabled=busy||offset===0;const next=button(t('next'),()=>{offset+=25;load();});next.disabled=busy||!data.has_more||offset>=100000;pager.append(previous,element('span',`${t('page')} ${offset/25+1}`),next);root.append(pager);
    }
    if(['overview','groups'].includes(view))root.append(element('p',t('note'),'host-note'));
  }
  function editRole(userId){window.SPBRoleEditor.open({request,language:lang,userId,onAuthError:fail,onSaved:async id=>{view='people';search=String(id);offset=0;filter='';await load();}});}
  async function prepareGroup(chatId){if(busy||closed)return;busy=true;error='';groupLink=null;render();try{const result=await request('host-group-link','POST',{chat_id:chatId});const url=new URL(result.url);if(url.protocol!=='https:'||url.hostname!=='t.me'||url.username||url.password)throw new Error('temporarily_unavailable');groupLink={chat_id:chatId,url:url.href};}catch(e){fail(e);}finally{busy=false;render();}}
  async function load(){if(busy||closed)return;busy=true;error='';data=null;groupLink=null;render();try{data=await request('host-query','POST',{view,search,filter,offset});}catch(e){fail(e);}finally{busy=false;render();}}
  function fail(e){error=Object.hasOwn(words.en,e.message)?e.message:'temporarily_unavailable';if(['session_expired','forbidden'].includes(error)){closed=true;data=null;clearToken?.();window.SPBRoleEditor.expire();window.SPBCasePanel.expire();}render();}
  async function logout(){if(busy||closed)return;busy=true;render();try{await request('logout','POST',{});closed=true;data=null;clearToken();error='signedOut';}catch(e){fail(e);}finally{busy=false;render();}}
  window.SPBHost={
    async start(api,language,clear){request=api;lang=language;clearToken=clear;document.body.classList.add('host-mode');root=document.querySelector('main');document.getElementById('save-bar').hidden=true;await load();},
    setLanguage(language){lang=language;render();window.SPBRoleEditor.setLanguage(language);window.SPBCasePanel.setLanguage(language);},fail
  };
})();
