(() => {
  'use strict';
  const words={
    'zh-Hant':{title:'管理權限',user:'用戶 ID',read:'讀取權限',role:'角色',maintainer:'維護員',reviewer:'審核員',enabled:'已授權',disabled:'未授權',grant:'授予權限',revoke:'撤銷權限',review:'查看變更',confirmGrant:'確認授權',confirmRevoke:'確認撤權',back:'返回修改',close:'關閉',reload:'重新讀取',loading:'正在讀取…',saving:'正在保存…',host:'主持人的身分固定，不能在這裡修改。',invalid:'請輸入有效的 Telegram 用戶 ID。',current:'目前權限',effectMaintainer:'維護員可以使用維護指令、管理審核員及處理舉報；不會取得主持人面板權限。',effectReviewer:'審核員可以處理舉報及訓練審核；不能修改全域設定或管理維護組。',stillReview:'此人仍是維護員，撤銷審核員角色後仍可審核。',loseMaintainer:'將失去維護指令權限。',keepReviewer:'仍保留審核員角色。',settings_changed:'權限已被其他操作修改。請重新讀取後再確認。',save_failed:'未能確認保存結果。可以重試同一次變更，或重新讀取目前權限。',temporarily_unavailable:'暫時無法讀取，請稍後再試。',rate_limited:'操作太頻繁，請稍候一分鐘。'},
    'zh-Hans':{title:'管理权限',user:'用户 ID',read:'读取权限',role:'角色',maintainer:'维护员',reviewer:'审核员',enabled:'已授权',disabled:'未授权',grant:'授予权限',revoke:'撤销权限',review:'查看变更',confirmGrant:'确认授权',confirmRevoke:'确认撤权',back:'返回修改',close:'关闭',reload:'重新读取',loading:'正在读取…',saving:'正在保存…',host:'主持人的身份固定，不能在这里修改。',invalid:'请输入有效的 Telegram 用户 ID。',current:'目前权限',effectMaintainer:'维护员可以使用维护指令、管理审核员及处理举报；不会取得主持人面板权限。',effectReviewer:'审核员可以处理举报及训练审核；不能修改全局设置或管理维护组。',stillReview:'此人仍是维护员，撤销审核员角色后仍可审核。',loseMaintainer:'将失去维护指令权限。',keepReviewer:'仍保留审核员角色。',settings_changed:'权限已被其他操作修改。请重新读取后再确认。',save_failed:'未能确认保存结果。可以重试同一次变更，或重新读取目前权限。',temporarily_unavailable:'暂时无法读取，请稍后重试。',rate_limited:'操作太频繁，请稍候一分钟。'},
    en:{title:'Manage roles',user:'User ID',read:'Load roles',role:'Role',maintainer:'Maintainer',reviewer:'Reviewer',enabled:'Granted',disabled:'Not granted',grant:'Grant role',revoke:'Revoke role',review:'Review change',confirmGrant:'Confirm grant',confirmRevoke:'Confirm revocation',back:'Back to editing',close:'Close',reload:'Reload',loading:'Loading…',saving:'Saving…',host:'The project host is fixed and cannot be changed here.',invalid:'Enter a valid Telegram user ID.',current:'Current roles',effectMaintainer:'Maintainers can use maintenance commands, manage reviewers and review reports. This does not grant access to the host panel.',effectReviewer:'Reviewers can review reports and training requests. They cannot change global settings or manage maintainers.',stillReview:'This person remains a maintainer and can still review after this reviewer role is revoked.',loseMaintainer:'Maintenance command access will be removed.',keepReviewer:'The reviewer role will remain.',settings_changed:'Roles changed in another operation. Reload before confirming.',save_failed:'Unable to confirm the save. Retry this change or reload the current roles.',temporarily_unavailable:'Unable to load. Please try again later.',rate_limited:'Too many requests. Please wait a minute.'}
  };
  let dialog,context,lang='zh-Hant',target='',snapshot=null,role='maintainer',enabled=false,pending=null,confirming=false,busy=false,conflict=false,error='';
  const t=key=>words[lang][key]||words[lang].temporarily_unavailable;
  const el=(tag,text)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;return node;};
  function button(label,action){const node=el('button',t(label));node.type='button';node.disabled=busy;node.addEventListener('click',action);return node;}
  function close(){if(!busy){dialog.close();snapshot=null;pending=null;}}
  function render(){
    if(!dialog?.open)return;
    dialog.replaceChildren();const heading=el('h2',t('title'));heading.id='role-heading';dialog.append(heading);
    const status=el('p',error?t(error):busy?t(confirming?'saving':'loading'):'');status.setAttribute('role',error?'alert':'status');status.className='host-error';dialog.append(status);
    if(confirming&&snapshot){
      dialog.append(el('p',`${t('user')}: ${snapshot.user_id}`),el('h3',t(role)),el('p',`${t(snapshot[role]?'enabled':'disabled')} → ${t(enabled?'enabled':'disabled')}`));
      dialog.append(el('p',t(role==='maintainer'?(enabled?'effectMaintainer':'loseMaintainer'):'effectReviewer')));
      if(role==='reviewer'&&!enabled&&snapshot.maintainer)dialog.append(el('p',t('stillReview')));
      if(role==='maintainer'&&!enabled&&snapshot.reviewer)dialog.append(el('p',t('keepReviewer')));
      const actions=el('div');actions.className='dialog-actions';
      if(error)actions.append(button('reload',read));
      const back=button('back',()=>{confirming=false;pending=null;error='';render();});back.disabled=busy||!!pending;
      const save=button(enabled?'confirmGrant':'confirmRevoke',saveRole);save.className='primary';save.disabled=busy||conflict;
      actions.append(back,save);dialog.append(actions);
    }else{
      const form=el('form');form.className='host-role-form';form.addEventListener('submit',event=>{event.preventDefault();read();});
      const label=el('label',t('user'));const input=el('input');input.type='text';input.inputMode='numeric';input.pattern='[0-9]+';input.maxLength=16;input.value=target;input.required=true;input.disabled=busy;
      input.addEventListener('input',()=>{target=input.value;snapshot=null;pending=null;conflict=false;dialog.querySelector('#role-controls')?.remove();});label.append(input);form.append(label,button('read',read));dialog.append(form);
      if(snapshot){
        const controls=el('div');controls.id='role-controls';
        controls.append(el('p',`${t('current')}: ${t('maintainer')} · ${t(snapshot.maintainer?'enabled':'disabled')} / ${t('reviewer')} · ${t(snapshot.reviewer?'enabled':'disabled')}`));
        if(snapshot.host){controls.append(el('p',t('host')));}else{
          const roleLabel=el('label',t('role'));const select=el('select');for(const name of ['maintainer','reviewer']){const option=el('option',t(name));option.value=name;select.append(option);}select.value=role;select.disabled=busy;select.addEventListener('change',()=>{role=select.value;enabled=snapshot[role];render();});roleLabel.append(select);
          const changeLabel=el('label',t('review'));const change=el('select');for(const [name,state] of [['grant',true],['revoke',false]]){const option=el('option',t(name));option.value=String(state);change.append(option);}change.value=String(enabled);change.disabled=busy;change.addEventListener('change',()=>{enabled=change.value==='true';render();});changeLabel.append(change);
          const review=button('review',()=>{confirming=true;error='';render();});review.disabled=busy||enabled===snapshot[role];controls.append(roleLabel,changeLabel,review);
        }
        dialog.append(controls);
      }
    }
    dialog.append(button('close',close));
  }
  function handleError(e){
    if(['session_expired','forbidden'].includes(e.message)){dialog.close();snapshot=null;pending=null;context.onAuthError(e);return;}
    error=Object.hasOwn(words.en,e.message)?e.message:'temporarily_unavailable';conflict=e.message==='settings_changed';
  }
  async function read(){
    if(busy)return;const id=Number(target);
    if(!/^[0-9]+$/.test(target)||!Number.isSafeInteger(id)||id<=0||id>=2**52){error='invalid';render();return;}
    busy=true;error='';render();
    try{snapshot=await context.request('host-role','POST',{user_id:id});pending=null;confirming=false;conflict=false;enabled=snapshot[role];}
    catch(e){handleError(e);}finally{busy=false;render();}
  }
  async function saveRole(){
    if(busy||conflict||!snapshot||snapshot.host)return;
    pending??={request_id:crypto.randomUUID(),user_id:snapshot.user_id,expected_revision:snapshot.revision,role,enabled};
    busy=true;error='';render();
    try{await context.request('host-role','PATCH',pending);const id=snapshot.user_id;pending=null;snapshot=null;dialog.close();await context.onSaved(id);}
    catch(e){handleError(e);}finally{busy=false;render();}
  }
  window.SPBRoleEditor={
    open(options){context=options;lang=options.language;target=options.userId?String(options.userId):'';snapshot=null;pending=null;confirming=false;busy=false;conflict=false;error='';role='maintainer';
      if(!dialog){dialog=el('dialog');dialog.className='host-role-dialog';dialog.setAttribute('aria-labelledby','role-heading');dialog.addEventListener('cancel',event=>{event.preventDefault();close();});document.body.append(dialog);}
      dialog.showModal();render();if(target)read();
    },
    setLanguage(language){lang=language;render();},
    back(){if(dialog?.open){close();return true;}return false;},
    expire(){if(dialog?.open)dialog.close();snapshot=null;pending=null;}
  };
})();
