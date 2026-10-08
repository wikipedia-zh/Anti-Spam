(() => {
  'use strict';
  const words={
    'zh-Hant':{title:'重試工作',kind:'工作種類',case_id:'案件 ID',chat_id:'群組 ID',attempts:'已嘗試',next_attempt_at:'原定重試時間',last_error:'上次錯誤',confirm:'確認重試',retry:'確認同一請求的結果',reload:'重新讀取',close:'關閉',loading:'處理中…',saved:'已排程重試，實際結果請查看工作記錄。',note:'會繼續未完成的步驟，並重新檢查權限及限制。',unknown:'未知',paused:'緊急控制已暫停這筆工作。',waiting:'群組無法存取，等待機器人重新加入或恢復存取。',noError:'這筆工作目前沒有待重試的錯誤。',cooldown:'Telegram 限流中，最早重試時間：',audit:'操作記錄',queue_item_missing:'工作已完成、取消或不存在。請重新讀取列表。',settings_changed:'工作狀態已改變，請重新讀取再確認。',queue_paused:'工作已暫停，請重新讀取狀態。',queue_busy:'工作正在執行或剛重試過，請稍後再試。',invalid_request:'無法重試這筆工作，請重新讀取。',temporarily_unavailable:'暫時無法讀取，請重試。',save_failed:'結果尚未確認，請重試同一請求。',pending:'上次重試的結果尚未確認，先確認下方這筆工作的結果。',rate_limited:'操作太頻繁，請稍後再試。'},
    en:{title:'Retry job',kind:'Job type',case_id:'Case ID',chat_id:'Group ID',attempts:'Attempts',next_attempt_at:'Scheduled retry',last_error:'Last error',confirm:'Confirm retry',retry:'Check the same request',reload:'Reload',close:'Close',loading:'Working…',saved:'Retry scheduled. Check the job record for its result.',note:'Continues unfinished steps and rechecks permissions and restrictions.',unknown:'Unknown',paused:'Emergency controls have paused this job.',waiting:'The group is inaccessible. Waiting for the bot to rejoin or access to return.',noError:'This job has no error to retry.',cooldown:'Telegram cooldown; earliest retry: ',audit:'Activity record',queue_item_missing:'The job completed, was cancelled or no longer exists. Refresh the list.',settings_changed:'The job changed. Reload before confirming.',queue_paused:'The job is paused. Reload its status.',queue_busy:'The job is running or was just retried. Try again shortly.',invalid_request:'This job cannot be retried. Please reload.',temporarily_unavailable:'Unable to load. Please retry.',save_failed:'The result is unconfirmed. Retry the same request.',pending:'A previous retry is unconfirmed. Check the job below first.',rate_limited:'Too many requests. Try again shortly.'}
  };
  words['zh-Hans']={...words['zh-Hant'],title:'重试工作',kind:'工作类型',case_id:'案件 ID',chat_id:'群组 ID',attempts:'已尝试',next_attempt_at:'原定重试时间',last_error:'上次错误',confirm:'确认重试',retry:'确认同一请求的结果',reload:'重新读取',close:'关闭',loading:'处理中…',saved:'已安排重试，实际结果请查看工作记录。',note:'会继续未完成的步骤，并重新检查权限及限制。',paused:'紧急控制已暂停这笔工作。',waiting:'群组无法访问，等待机器人重新加入或恢复访问。',noError:'这笔工作目前没有待重试的错误。',cooldown:'Telegram 限流中，最早重试时间：',audit:'操作记录',queue_item_missing:'工作已完成、取消或不存在。请重新读取列表。',settings_changed:'工作状态已改变，请重新读取再确认。',queue_paused:'工作已暂停，请重新读取状态。',queue_busy:'工作正在执行或刚重试过，请稍后再试。',invalid_request:'无法重试这笔工作，请重新读取。',temporarily_unavailable:'暂时无法读取，请重试。',save_failed:'结果尚未确认，请重试同一请求。',pending:'上次重试的结果尚未确认，先确认下方这笔工作的结果。',rate_limited:'操作太频繁，请稍后重试。'};
  Object.assign(words['zh-Hant'],{message_id:'訊息 ID',user_id:'對象 ID',request_id:'退群請求 ID',id:'通知 ID'});Object.assign(words['zh-Hans'],{message_id:'消息 ID',user_id:'对象 ID',request_id:'退群请求 ID',id:'通知 ID'});Object.assign(words.en,{message_id:'Message ID',user_id:'Target ID',request_id:'Departure request ID',id:'Notice ID'});
  Object.assign(words['zh-Hant'],{dependency:'等待這個群組的聯防封禁完成。',reversal:'會重試該案件尚未完成的解封；各群仍會核對其他有效封禁。'});Object.assign(words['zh-Hans'],{dependency:'等待这个群组的联防封禁完成。',reversal:'会重试该案件尚未完成的解封；各群仍会核对其他有效封禁。'});Object.assign(words.en,{dependency:'Waiting for this group’s shared ban to complete.',reversal:'Retries this case’s unfinished reversals. Other active bans are still checked in each group.'});
  let context,lang='zh-Hant',dialog,target,snapshot,pending=null,result=null,busy=false,error='',conflict=false;
  const t=k=>words[lang][k]??k;
  const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
  const date=v=>v?new Date(v*1000).toLocaleString(lang):t('unknown');
  function button(key,fn){const n=el('button',t(key));n.type='button';n.disabled=busy;n.addEventListener('click',fn);return n;}
  function close(){if(!busy)dialog?.close();}
  function render(){
    if(!dialog?.open)return;dialog.replaceChildren();const h=el('h2',t('title'));h.id='queue-heading';dialog.append(h);
    const status=el('p',error?t(error):busy?t('loading'):result?t('saved'):'',error?'host-error':'');status.setAttribute('role',error?'alert':'status');dialog.append(status);
    if(result)dialog.append(el('p',`${t('audit')} #${result.action_id}`));
    else if(snapshot){
      const dl=el('dl');for(const key of ['kind','case_id','chat_id','attempts','next_attempt_at'])if(snapshot[key]!=null)dl.append(el('dt',t(key)),el('dd',key==='next_attempt_at'?date(snapshot[key]):String(snapshot[key])));for(const key of ['message_id','user_id','request_id','id'])if(snapshot.target[key]!=null)dl.append(el('dt',t(key)),el('dd',String(snapshot.target[key])));dialog.append(dl);
      if(snapshot.last_error)dialog.append(el('h3',t('last_error')),el('pre',snapshot.last_error));
      dialog.append(el('p',t(pending?'pending':'note'),'host-note'));
      if(snapshot.target.kind==='reversal')dialog.append(el('p',t('reversal'),'host-note'));
      if(snapshot.waiting_for_group)dialog.append(el('p',t('waiting'),'host-note'));
      else if(snapshot.paused)dialog.append(el('p',t('paused'),'host-note'));
      else if(snapshot.waiting_for_dependency)dialog.append(el('p',t('dependency'),'host-note'));
      else if(!snapshot.retryable)dialog.append(el('p',t('noError'),'host-note'));
      if(snapshot.telegram_not_before>Date.now()/1000)dialog.append(el('p',t('cooldown')+date(snapshot.telegram_not_before),'host-note'));
      const retry=button(pending?'retry':'confirm',save);retry.disabled=busy||conflict||!snapshot.retryable;retry.className='primary';dialog.append(retry);
    }
    const actions=el('div',undefined,'dialog-actions');const reload=button('reload',read);reload.disabled=busy||!!pending;actions.append(reload,button('close',close));dialog.append(actions);
  }
  function fail(e){if(['session_expired','forbidden'].includes(e.message)){window.SPBQueue.expire();context.onAuthError(e);return;}error=Object.hasOwn(words.en,e.message)?e.message:'temporarily_unavailable';if(['settings_changed','invalid_request','queue_item_missing','queue_paused'].includes(error)){pending=null;conflict=true;}}
  async function read(){if(busy||pending)return;busy=true;error='';snapshot=null;result=null;conflict=false;render();try{snapshot=await context.request('host-queue-item','POST',{target});}catch(e){fail(e);}finally{busy=false;render();}}
  async function save(){if(busy||conflict||!snapshot?.retryable)return;pending??={request_id:crypto.randomUUID(),target,expected_revision:snapshot.revision};busy=true;error='';render();try{result=await context.request('host-queue-item','PATCH',pending);pending=null;await context.onSaved();}catch(e){fail(e);}finally{busy=false;render();}}
  window.SPBQueue={open(opts){if(busy)return;context=opts;lang=opts.language;if(!dialog){dialog=el('dialog',undefined,'host-rebuild-dialog');dialog.setAttribute('aria-labelledby','queue-heading');dialog.addEventListener('cancel',e=>{e.preventDefault();close();});document.body.append(dialog);}dialog.showModal();if(pending){error='save_failed';render();}else{target=opts.target;read();}},setLanguage(value){lang=value;render();},back(){if(dialog?.open){close();return true;}return false;},expire(){dialog?.close();target=null;snapshot=null;pending=null;result=null;}};
})();
