(() => {
  'use strict';
  const words={
    'zh-Hant':{title:'規則管理',name:'名稱',pattern:'正則',sample:'測試文字',test:'測試正則',review:'查看變更',remove:'刪除規則',confirm:'確認保存',confirmRemove:'確認刪除',back:'返回修改',close:'關閉',reload:'重新讀取',loading:'正在處理…',saved:'已保存',before:'修改前',after:'修改後',none:'無',missing:'規則已刪除，請返回清單。',testNote:'只測試這段文字，不會封禁或訓練。正式規則也會檢查用戶名稱等內容。',impact:'保存後會影響所有使用此規則的群組。既有封禁不會自動撤銷。',deleteImpact:'刪除後停止使用這條規則，既有封禁仍保留。',match:'匹配成功',no_match:'沒有匹配',invalid:'正則無效，請檢查語法。',limit:'超出測試運算限制，請簡化正則。',empty:'（空字串）',needsTest:'修改正則後，請先測試再確認。',settings_changed:'規則已被其他操作修改。請重新讀取後確認。',invalid_request:'請檢查正則與欄位長度。',save_failed:'未能確認保存結果。可重試同一請求，或重新讀取。',temporarily_unavailable:'暫時無法處理，請稍後再試。',rate_limited:'測試或請求繁忙，請稍後再試。'},
    'zh-Hans':{title:'规则管理',name:'名称',pattern:'正则',sample:'测试文字',test:'测试正则',review:'查看变更',remove:'删除规则',confirm:'确认保存',confirmRemove:'确认删除',back:'返回修改',close:'关闭',reload:'重新读取',loading:'正在处理…',saved:'已保存',before:'修改前',after:'修改后',none:'无',missing:'规则已删除，请返回列表。',testNote:'只测试这段文字，不会封禁或训练。正式规则也会检查用户名称等内容。',impact:'保存后会影响所有使用此规则的群组。既有封禁不会自动撤销。',deleteImpact:'删除后停止使用这条规则，既有封禁仍保留。',match:'匹配成功',no_match:'没有匹配',invalid:'正则无效，请检查语法。',limit:'超出测试运算限制，请简化正则。',empty:'（空字符串）',needsTest:'修改正则后，请先测试再确认。',settings_changed:'规则已被其他操作修改。请重新读取后确认。',invalid_request:'请检查正则和字段长度。',save_failed:'未能确认保存结果。可重试同一请求，或重新读取。',temporarily_unavailable:'暂时无法处理，请稍后重试。',rate_limited:'测试或请求繁忙，请稍后重试。'},
    en:{title:'Manage rule',name:'Name',pattern:'Pattern',sample:'Test text',test:'Test pattern',review:'Review changes',remove:'Delete rule',confirm:'Confirm save',confirmRemove:'Confirm deletion',back:'Back to editing',close:'Close',reload:'Reload',loading:'Working…',saved:'Saved',before:'Before',after:'After',none:'None',missing:'This rule was deleted. Return to the list.',testNote:'Tests this text only. It does not ban or train. Live rules also check content such as display names.',impact:'Saving affects all groups using this rule. Existing bans are not automatically reversed.',deleteImpact:'Stops using this rule. Existing bans remain.',match:'Match found',no_match:'No match',invalid:'Invalid pattern. Check its syntax.',limit:'Test computation limit reached. Simplify the pattern.',empty:'(empty string)',needsTest:'Test the edited pattern before confirming.',settings_changed:'Rules changed in another operation. Reload before confirming.',invalid_request:'Check the pattern and field lengths.',save_failed:'Unable to confirm the save. Retry this request or reload.',temporarily_unavailable:'Unable to process the request. Try again later.',rate_limited:'Tests or requests are busy. Try again shortly.'}
  };
  let dialog,context,lang='zh-Hant',snapshot=null,pattern='',description='',sample='',tested=null,pending=null,confirming='',busy=false,conflict=false,error='',saved=false;
  const t=key=>words[lang][key]||words[lang].temporarily_unavailable;
  const el=(tag,text,className)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n;};
  function button(key,action){const n=el('button',t(key));n.type='button';n.disabled=busy;n.addEventListener('click',action);return n;}
  function close(){if(!busy){dialog.close();snapshot=null;pending=null;}}
  function validTest(){return tested?.pattern===pattern&&tested?.text===sample&&['match','no_match'].includes(tested.result.status);}
  function changed(){return snapshot&&(snapshot.rule?.pattern!==pattern||snapshot.rule?.description!==description);}
  function syncReview(){const b=dialog.querySelector('[data-rule-review]');if(b)b.disabled=busy||!changed()||!validTest();}
  function field(key,tag,value,max,onInput){const label=el('label',t(key));const n=el(tag);n.value=value;n.maxLength=max;n.disabled=busy;if(tag==='textarea')n.rows=key==='sample'?5:3;n.addEventListener('input',()=>{onInput(n.value);saved=false;syncReview();});label.append(n);return label;}
  function showRule(label,rule){const box=el('article',undefined,'host-record');box.append(el('h3',t(label)));if(rule){box.append(el('p',rule.description||t('none')),el('pre',rule.pattern));}else box.append(el('p',t('none')));return box;}
  function render(){
    if(!dialog?.open)return;dialog.replaceChildren();const heading=el('h2',t('title'));heading.id='rule-heading';dialog.append(heading);
    const status=el('p',error?t(error):busy?t('loading'):saved?`${t('saved')} · @${snapshot?.rule_id}`:'');status.setAttribute('role',error?'alert':'status');if(error)status.className='host-error';dialog.append(status);
    if(snapshot){
      if(snapshot.rule_id!==null&&!snapshot.rule){dialog.append(el('p',t('missing')));}
      else if(confirming){
        dialog.append(showRule('before',snapshot.rule),showRule('after',confirming==='delete'?null:{pattern,description}),el('p',t(confirming==='delete'?'deleteImpact':'impact'),'host-note'));
        const actions=el('div',undefined,'dialog-actions');const back=button('back',()=>{confirming='';error='';render();});back.disabled=busy||!!pending;
        const save=button(confirming==='delete'?'confirmRemove':'confirm',saveRule);save.disabled=busy||conflict;save.className='primary';actions.append(back,save);dialog.append(actions);
      }else{
        const form=el('div',undefined,'host-rule-form');
        const invalidate=()=>{tested=null;dialog.querySelector('#rule-result')?.remove();};
        form.append(field('name','input',description,256,v=>description=v),field('pattern','textarea',pattern,512,v=>{pattern=v;invalidate();}),field('sample','textarea',sample,4000,v=>{sample=v;invalidate();}));
        form.append(el('p',t('testNote'),'host-note'),button('test',testRule));dialog.append(form);
        if(tested){const result=el('article',undefined,'host-record');result.id='rule-result';result.setAttribute('role','status');result.append(el('p',t(tested.result.status)));if(tested.result.status==='match')result.append(el('pre',tested.result.matched||t('empty')));dialog.append(result);}
        const actions=el('div',undefined,'dialog-actions');const review=button('review',()=>{if(!validTest()){error='needsTest';render();return;}confirming='save';error='';render();dialog.scrollTop=0;});review.dataset.ruleReview='';review.disabled=busy||!changed()||!validTest();actions.append(review);
        if(snapshot.rule)actions.append(button('remove',()=>{confirming='delete';error='';render();dialog.scrollTop=0;}));dialog.append(actions);
      }
    }
    const actions=el('div',undefined,'dialog-actions');actions.append(button('reload',read),button('close',close));dialog.append(actions);
  }
  function fail(e){if(['session_expired','forbidden'].includes(e.message)){dialog.close();snapshot=null;pending=null;context.onAuthError(e);return;}error=Object.hasOwn(words.en,e.message)?e.message:'temporarily_unavailable';conflict=['settings_changed','invalid_request'].includes(e.message);}
  async function read(){if(busy)return;busy=true;error='';render();try{snapshot=await context.request('host-rule','POST',{rule_id:context.ruleId??null});pattern=snapshot.rule?.pattern||'';description=snapshot.rule?.description||'';tested=null;pending=null;confirming='';conflict=false;saved=false;}catch(e){fail(e);}finally{busy=false;render();}}
  async function testRule(){if(busy||!pattern.trim())return;busy=true;error='';tested=null;render();try{const result=await context.request('host-rule-test','POST',{pattern,text:sample});tested={pattern,text:sample,result};}catch(e){fail(e);}finally{busy=false;render();}}
  async function saveRule(){
    if(busy||conflict||!snapshot)return;
    pending??={request_id:crypto.randomUUID(),rule_id:snapshot.rule_id,expected_revision:snapshot.revision,rule:confirming==='delete'?null:{pattern,description}};
    busy=true;error='';saved=false;render();
    try{snapshot=await context.request('host-rule','PATCH',pending);context.ruleId=snapshot.rule_id;pending=null;confirming='';saved=true;tested=null;pattern=snapshot.rule?.pattern||'';description=snapshot.rule?.description||'';await context.onSaved();}catch(e){fail(e);}finally{busy=false;render();}
  }
  window.SPBRuleEditor={
    open(options){context=options;lang=options.language;snapshot=null;pattern='';description='';sample='';tested=null;pending=null;confirming='';busy=false;conflict=false;error='';saved=false;if(!dialog){dialog=el('dialog',undefined,'host-rule-dialog');dialog.setAttribute('aria-labelledby','rule-heading');dialog.addEventListener('cancel',event=>{event.preventDefault();close();});document.body.append(dialog);}dialog.showModal();render();read();},
    setLanguage(value){lang=value;render();},back(){if(dialog?.open){close();return true;}return false;},expire(){if(dialog?.open)dialog.close();snapshot=null;pending=null;}
  };
})();
