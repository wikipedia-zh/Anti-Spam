(() => {
  'use strict';
  const words={
    'zh-Hant':{title:'案件詳情',reverse:'撤銷這宗封禁',confirm:'確認撤銷',back:'返回',close:'關閉',reload:'重新讀取',loading:'正在讀取…',saving:'正在保存撤銷請求…',queued:'撤銷請求已保存。解封失敗會自動重試，請重新讀取查看進度。',case:'案件',user:'用戶',group:'來源群組',status:'目前狀態',reason:'觸發原因',score:'模型分數',evidence:'保存的證據',threshold:'這宗案件沒有保存當時的判斷門檻。',impact:'撤銷範圍',unban_count:'預計解封的群組',retained_count:'因其他有效案件而保留',cancel_count:'取消待送聯防',training_samples:'移除本案訓練樣本',notice:'只撤銷這宗案件。其他有效案件仍可能使此用戶被封禁；執行時會重新檢查。',delivery:'各群處理情況',previous:'上一頁',next:'下一頁',unknown:'未知',origin:'來源群組',pending:'待送',done:'已完成封禁',cancelled:'已取消',unban:'解封／重試解封',retain:'保留其他案件的封禁',cancel:'取消待送封禁',none:'沒有待解封項目',uncertain:'Telegram 結果未能確認',readOnly:'此案件目前沒有可提交的封禁撤銷操作。',settings_changed:'案件或影響範圍已改變。請重新讀取並確認。',case_not_found:'找不到這宗案件。',save_failed:'未能確認保存結果。可以重試同一請求，或重新讀取案件。',invalid_request:'這宗案件目前不能執行此操作，請重新讀取。',temporarily_unavailable:'暫時無法讀取，請稍後再試。',rate_limited:'操作太頻繁，請稍候一分鐘再試。'},
    'zh-Hans':{title:'案件详情',reverse:'撤销这宗封禁',confirm:'确认撤销',back:'返回',close:'关闭',reload:'重新读取',loading:'正在读取…',saving:'正在保存撤销请求…',queued:'撤销请求已保存。解封失败会自动重试，请重新读取查看进度。',case:'案件',user:'用户',group:'来源群组',status:'当前状态',reason:'触发原因',score:'模型分数',evidence:'保存的证据',threshold:'这宗案件没有保存当时的判断阈值。',impact:'撤销范围',unban_count:'预计解封的群组',retained_count:'因其他有效案件而保留',cancel_count:'取消待发联防',training_samples:'移除本案训练样本',notice:'只撤销这宗案件。其他有效案件仍可能使此用户被封禁；执行时会重新检查。',delivery:'各群处理情况',previous:'上一页',next:'下一页',unknown:'未知',origin:'来源群组',pending:'待发',done:'已完成封禁',cancelled:'已取消',unban:'解封／重试解封',retain:'保留其他案件的封禁',cancel:'取消待发封禁',none:'没有待解封项目',uncertain:'Telegram 结果未能确认',readOnly:'此案件当前没有可提交的封禁撤销操作。',settings_changed:'案件或影响范围已改变。请重新读取并确认。',case_not_found:'找不到这宗案件。',save_failed:'未能确认保存结果。可以重试同一请求，或重新读取案件。',invalid_request:'这宗案件当前不能执行此操作，请重新读取。',temporarily_unavailable:'暂时无法读取，请稍后再试。',rate_limited:'操作太频繁，请稍候一分钟再试。'},
    en:{title:'Case details',reverse:'Reverse this ban',confirm:'Confirm reversal',back:'Back',close:'Close',reload:'Refresh case',loading:'Loading…',saving:'Saving reversal request…',queued:'Reversal saved. Failed unbans will retry automatically. Refresh to check progress.',case:'Case',user:'User',group:'Source group',status:'Current status',reason:'Reason',score:'Model score',evidence:'Saved evidence',threshold:'The decision threshold was not recorded for this case.',impact:'Reversal scope',unban_count:'Groups to unban',retained_count:'Kept due to other active cases',cancel_count:'Pending deliveries to cancel',training_samples:'Case training samples to remove',notice:'This reverses only this case. Other active cases may still ban this user; they are checked again during execution.',delivery:'Group outcomes',previous:'Previous',next:'Next',unknown:'Unknown',origin:'Source group',pending:'Pending delivery',done:'Ban delivered',cancelled:'Cancelled',unban:'Unban / retry unban',retain:'Keep the ban from another case',cancel:'Cancel pending ban',none:'No remaining unban',uncertain:'Telegram outcome unconfirmed',readOnly:'No ban reversal can be submitted for this case in its current state.',settings_changed:'The case or its impact has changed. Refresh and review it again.',case_not_found:'Case not found.',save_failed:'The save could not be confirmed. Retry the same request or refresh the case.',invalid_request:'This action is not available for the case. Refresh to check its state.',temporarily_unavailable:'Unable to load. Please try again.',rate_limited:'Too many requests. Please wait a minute.'}
  };
  Object.assign(words['zh-Hant'],{review:'審核',reviewConfirm:'確認審核決定',approve:'批准',reject:'拒絕',approveReport:'受理並封禁',rejectReport:'拒絕並標記正常',approveTraining:'訓練並加入聯防',rejectTraining:'拒絕訓練',reviewSaved:'審核決定已保存。請重新讀取查看執行進度。',saving:'正在保存請求…',reportApprove:'受理後會排程封禁來源群組；成功後訓練為垃圾訊息，並依聯防設定送至訂閱群組。',reportReject:'內容會標記為正常。非維護員的舉報者會增加一次被拒記錄，達到上限後暫停使用 /spam。',trainingApprove:'內容會訓練為垃圾訊息，並加入跨群黑名單。其他群組依聯防設定接收封禁。',trainingReject:'不訓練、不因這次審核加入聯防；原有的本群封禁保留。',reporter:'舉報者',strikes:'被拒次數',exempt:'此舉報者不計被拒次數。',missingCard:'未保存原審核訊息位置，舊訊息可能不會更新。請以此處的審核結果為準。',decided:'已保存的決定',reviewer:'審核人',confirmReview:'確認決定'});
  Object.assign(words['zh-Hans'],{review:'审核',reviewConfirm:'确认审核决定',approve:'批准',reject:'拒绝',approveReport:'受理并封禁',rejectReport:'拒绝并标记正常',approveTraining:'训练并加入联防',rejectTraining:'拒绝训练',reviewSaved:'审核决定已保存。请重新读取查看执行进度。',saving:'正在保存请求…',reportApprove:'受理后会安排封禁来源群组；成功后训练为垃圾消息，并按联防设置发送至订阅群组。',reportReject:'内容会标记为正常。非维护员的举报者会增加一次被拒记录，达到上限后暂停使用 /spam。',trainingApprove:'内容会训练为垃圾消息，并加入跨群黑名单。其他群组按联防设置接收封禁。',trainingReject:'不训练、不因本次审核加入联防；原有的本群封禁保留。',reporter:'举报者',strikes:'被拒次数',exempt:'此举报者不计被拒次数。',missingCard:'未保存原审核消息位置，旧消息可能不会更新。请以此处的审核结果为准。',decided:'已保存的决定',reviewer:'审核人',confirmReview:'确认决定'});
  Object.assign(words.en,{review:'Review',reviewConfirm:'Confirm review decision',approve:'Approve',reject:'Reject',approveReport:'Accept and ban',rejectReport:'Reject and mark as normal',approveTraining:'Train and share ban',rejectTraining:'Reject training',reviewSaved:'Decision saved. Refresh to check execution progress.',saving:'Saving request…',reportApprove:'Queues a ban in the source group. After it succeeds, the content trains the spam model and the ban is shared with subscribed groups.',reportReject:'Labels the content as normal. Reporters who are not maintainers receive one rejection strike; reaching the limit suspends /spam access.',trainingApprove:'Trains the content as spam and adds the user to the shared blacklist. Other groups receive bans according to their sharing settings.',trainingReject:'Does not train or add a shared ban through this review. The existing local ban remains.',reporter:'Reporter',strikes:'Rejection strikes',exempt:'This reporter is exempt from rejection strikes.',missingCard:'The original review message location was not saved, so the old message may not update. Use the decision shown here.',decided:'Saved decision',reviewer:'Reviewer',confirmReview:'Confirm decision'});
  let dialog,context,lang='zh-Hant',data=null,busy=false,error='',confirming='',pending=null,conflict=false,saved=false,savedMessage='queued';
  const t=key=>words[lang][key]||key;
  const el=(tag,text,className)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;};
  function button(label,fn){const node=el('button',t(label));node.type='button';node.disabled=busy;node.addEventListener('click',fn);return node;}
  function close(){if(!busy){dialog.close();data=null;pending=null;}}
  function render(){
    if(!dialog?.open)return;
    const reviewing=['approve','reject'].includes(confirming);
    dialog.replaceChildren();const heading=el('h2',t(reviewing?'reviewConfirm':confirming?'reverse':'title'));heading.id='case-heading';dialog.append(heading);
    const status=el('p',error?t(error):busy?t(pending?'saving':'loading'):saved?t(savedMessage):'');status.setAttribute('role',error?'alert':'status');if(error)status.className='host-error';dialog.append(status);
    if(data){
      const c=data.case;const info=el('article',undefined,'host-record');const dl=el('dl');
      for(const [label,value] of [['case',c.id],['user',`${c.target_user_id} · ${c.target_name}`],['group',c.chat_id],['status',context.status(c.status)]])dl.append(el('dt',t(label)),el('dd',String(value)));
      info.append(dl);dialog.append(info);
      if(!confirming){
        const evidence=el('details');evidence.append(el('summary',t('evidence')),el('pre',c.evidence||t('unknown')));dialog.append(evidence);
        dialog.append(el('p',`${t('reason')}: ${c.reason||t('unknown')}`));
        if(c.model_score!==null){dialog.append(el('p',`${t('score')}: ${c.model_score}`),el('p',t('threshold'),'host-note'));}
      }
      const review=data.review;
      if(review?.kind&&confirming!=='reverse'){
        dialog.append(el('h3',t('review')));
        if(review.decision)dialog.append(el('p',`${t('decided')}: ${t(review.decision)} · ${t('reviewer')}: ${review.actor_id??t('unknown')}`));
        if(reviewing)dialog.append(el('p',t(review.kind+(confirming==='approve'?'Approve':'Reject'))));
        if(review.reporter_id!==null&&review.reporter_id!==undefined){dialog.append(el('p',`${t('reporter')}: ${review.reporter_id}`),el('p',review.reporter_exempt?t('exempt'):`${t('strikes')}: ${review.reporter_strikes} / ${review.strike_limit}`));}
        if(!review.message_known)dialog.append(el('p',t('missingCard'),'host-note'));
      }
      if(confirming==='reverse'||['reversal_pending','reversed'].includes(c.status)){
        dialog.append(el('h3',t('impact')));const impact=el('dl',undefined,'case-impact');
        for(const key of ['unban_count','retained_count','cancel_count','training_samples'])impact.append(el('dt',t(key)),el('dd',String(data[key])));
        dialog.append(impact,el('p',t('notice'),'host-note'));
      }
      if(!reviewing&&(data.can_reverse||!review?.can_review)){dialog.append(el('h3',t('delivery')));
      for(const target of data.targets){
        const row=el('article',undefined,'host-record');row.append(el('h4',target.title||String(target.chat_id)));if(target.title)row.append(el('p',String(target.chat_id),'host-meta'));
        const parts=[target.origin?t('origin'):t(target.state||'unknown')];
        if(target.outcome_unknown)parts.push(t('uncertain'));
        if(confirming==='reverse'||c.status==='reversal_pending')parts.push(t(target.reversal_action));
        row.append(el('p',parts.join(' · ')));
        if(target.last_error)row.append(el('pre',target.last_error));dialog.append(row);
      }
      if(!confirming){
        const pages=el('div',undefined,'dialog-actions');const prev=button('previous',()=>read(data.offset-25));prev.disabled=busy||data.offset===0;const next=button('next',()=>read(data.offset+25));next.disabled=busy||!data.has_more;pages.append(prev,el('span',`${data.total?data.offset+1:0}–${Math.min(data.offset+25,data.total)} / ${data.total}`),next);dialog.append(pages);
      }}
      const actions=el('div',undefined,'dialog-actions');
      if(confirming){const back=button('back',()=>{confirming='';error='';render();});back.disabled=busy||!!pending;const save=button(reviewing?'confirmReview':'confirm',saveAction);save.className='primary';save.disabled=busy||conflict;actions.append(back,save);}
      else {
        const confirm=action=>{confirming=action;error='';render();dialog.scrollTop=0;};
        if(review?.can_review){actions.append(button(review.kind==='report'?'approveReport':'approveTraining',()=>confirm('approve')),button(review.kind==='report'?'rejectReport':'rejectTraining',()=>confirm('reject')));}
        if(data.can_reverse)actions.append(button('reverse',()=>confirm('reverse')));
        else if(!saved&&!review?.kind)dialog.append(el('p',t('readOnly'),'host-note'));
      }
      dialog.append(actions);
    }
    const actions=el('div',undefined,'dialog-actions');actions.append(button('reload',()=>read(0)),button('close',close));dialog.append(actions);
  }
  function fail(e){
    if(['session_expired','forbidden'].includes(e.message)){dialog.close();data=null;pending=null;context.onAuthError(e);return;}
    error=Object.hasOwn(words.en,e.message)?e.message:'temporarily_unavailable';conflict=['settings_changed','invalid_request','case_not_found'].includes(e.message);
  }
  async function read(offset=0){if(busy)return;busy=true;error='';render();try{data=await context.request('host-case','POST',{case_id:context.caseId,offset});pending=null;confirming='';conflict=false;}catch(e){fail(e);}finally{busy=false;render();}}
  async function saveAction(){
    const reversing=confirming==='reverse';
    if(busy||conflict||(reversing?!data?.can_reverse:!data?.review?.can_review))return;
    pending??={request_id:crypto.randomUUID(),case_id:context.caseId,target_user_id:data.case.target_user_id,expected_revision:data.revision,...(reversing?{}:{kind:data.review.kind,decision:confirming})};busy=true;error='';saved=false;render();
    try{await context.request(reversing?'host-case-reverse':'host-case-review','PATCH',pending);pending=null;confirming='';saved=true;savedMessage=reversing?'queued':'reviewSaved';data=null;await context.onSaved();}
    catch(e){fail(e);}finally{busy=false;render();}
    if(saved&&dialog.open)await read(0);
  }
  window.SPBCasePanel={
    open(options){context=options;lang=options.language;data=null;busy=false;error='';confirming='';pending=null;conflict=false;saved=false;savedMessage='queued';
      if(!dialog){dialog=el('dialog',undefined,'host-case-dialog');dialog.setAttribute('aria-labelledby','case-heading');dialog.addEventListener('cancel',event=>{event.preventDefault();close();});document.body.append(dialog);}
      dialog.showModal();render();read();
    },
    setLanguage(language){lang=language;render();},
    back(){if(dialog?.open){close();return true;}return false;},
    expire(){if(dialog?.open)dialog.close();data=null;pending=null;}
  };
})();
