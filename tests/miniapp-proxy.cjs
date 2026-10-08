const {test, before, after} = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const {spawn} = require('node:child_process');
const path = require('node:path');

let upstream, php, port, seen = [], responseStatus = 200, responseBody = '{}', stopped;
const origin = 'https://spb.example';
const key = 'isolated-test-proxy-key-32-characters';
const bearer = `Bearer ${'a'.repeat(32)}`;
const listen = server => new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server.address().port)));
before(async () => {
  upstream = http.createServer(async (req, res) => {
    let body = ''; for await (const chunk of req) body += chunk;
    seen.push({path:req.url, method:req.method, headers:req.headers, body});
    res.writeHead(responseStatus, {'Content-Type':'application/json'}); res.end(responseBody);
  });
  const upstreamPort = await listen(upstream);
  const reservation = http.createServer(); port = await listen(reservation);
  await new Promise(resolve => reservation.close(resolve));
  php = spawn(process.env.PHP_BIN || 'php', ['-S', `127.0.0.1:${port}`, '-t', path.join(__dirname,'..','web','settings')], {
    env:{...process.env,MINIAPP_ORIGIN:origin,MINIAPP_PROXY_KEY:key,MINIAPP_UPSTREAM:`http://127.0.0.1:${upstreamPort}`},stdio:'ignore',windowsHide:true
  });
  stopped = new Promise(resolve => php.once('exit', resolve));
  for (let i=0;i<100;i++) {
    if (php.exitCode !== null) throw new Error('PHP test server stopped');
    try { await fetch(`http://127.0.0.1:${port}/api.php`); return; } catch {}
    await new Promise(resolve => setTimeout(resolve,50));
  }
  throw new Error('PHP test server did not start');
});
after(async () => { if(php) {php.kill(); await stopped;} if(upstream) await new Promise(resolve => upstream.close(resolve)); });
function request(route='settings', method='GET', headers={}, body) {
  return fetch(`http://127.0.0.1:${port}/api.php?route=${route}`,{method,headers:{Origin:origin,Authorization:bearer,'Content-Type':'application/json',...headers},...(body===undefined?{}:{body})});
}
test('only configured routes and same-origin requests reach the backend', async () => {
  for (const response of [
    await request('settings','GET',{Origin:'https://other.example'}),
    await request('settings','GET',{'Sec-Fetch-Site':'cross-site'}),
    await request('settings','PATCH',{Origin:''},'{}'),
    await request('settings','GET',{Authorization:''}),
    await request('http://other.example/'),
    await request('session','GET'),
    await request('settings','POST',{},'{}'),
    await request('settings','PATCH',{'Content-Type':'text/plain'},'{}'),
    await request('settings','PATCH',{},'x'.repeat(20481))
  ]) assert.ok([400,401,403,404,405,413,415].includes(response.status));
  assert.equal(seen.length,0);
});
test('the proxy forwards fixed paths, exact JSON, and its own key', async () => {
  const payload = JSON.stringify({request_id:'test',expected_revision:2,changes:{captcha:true}});
  const result = await request('settings','PATCH',{'X-SPB-Proxy-Key':'client-supplied'},payload);
  assert.equal(result.status,200);
  assert.equal(result.headers.get('cache-control'),'no-store');
  const last=seen.at(-1);
  assert.equal(last.path,'/api/groups/current/settings');
  assert.equal(last.method,'PATCH'); assert.equal(last.body,payload);
  assert.equal(last.headers['x-spb-proxy-key'],key);
  assert.equal(last.headers.authorization,bearer); assert.equal(last.headers.origin,origin);
  const get=await request('settings','GET',{Origin:'','Sec-Fetch-Site':'same-origin'});
  assert.equal(get.status,200);
  const login=await request('session','POST',{},JSON.stringify({init_data:'local-test'}));
  assert.equal(login.status,200); assert.equal(seen.at(-1).path,'/api/miniapp/session');
  assert.equal(seen.at(-1).headers.authorization,undefined);
});
test('management queries use fixed authenticated POST routes', async () => {
  for (const [route,path] of [['host-query','/api/host/query'],['logout','/api/miniapp/logout'],['host-group-link','/api/host/group-link'],['host-case','/api/host/case'],['host-rule-test','/api/host/rule/test'],['host-model','/api/host/model']]) {
    const count=seen.length;
    assert.equal((await request(route,'GET')).status,405);
    assert.equal((await request(route,'POST',{Authorization:''},'{}')).status,401);
    assert.equal((await request(route,'POST',{Origin:'https://other.example'},'{}')).status,403);
    assert.equal(seen.length,count);
    assert.equal((await request(route,'POST',{},JSON.stringify({view:'overview'}))).status,200);
    assert.equal(seen.at(-1).path,path);
    assert.equal(seen.at(-1).headers.authorization,bearer);
    assert.equal(seen.at(-1).headers['x-spb-proxy-key'],key);
  }
});
test('role reads and changes require authentication and fixed methods', async () => {
  for(const [route,path] of [['host-role','/api/host/role'],['host-rule','/api/host/rule']]) {
  const before=seen.length;
  assert.equal((await request(route,'GET')).status,405);
  assert.equal((await request(route,'PATCH',{Authorization:''},'{}')).status,401);
  assert.equal((await request(route,'POST',{Origin:'https://other.example'},'{}')).status,403);
  assert.equal(seen.length,before);
  for(const method of ['POST','PATCH']){
    const payload=JSON.stringify({user_id:300,role:'maintainer',enabled:true});
    assert.equal((await request(route,method,{},payload)).status,200);
    assert.equal(seen.at(-1).path,path);assert.equal(seen.at(-1).body,payload);
  }
  }
});
test('conflicts survive the proxy and upstream failures stay private', async () => {
  const count=seen.length;
  for (const route of ['host-case-reverse','host-case-review']) {
    assert.equal((await request(route,'POST',{},'{}')).status,405);
    assert.equal((await request(route,'PATCH',{Authorization:''},'{}')).status,401);
    assert.equal((await request(route,'PATCH',{Origin:'https://other.example'},'{}')).status,403);
  }
  assert.equal(seen.length,count);
  assert.equal((await request('host-case-reverse','PATCH',{},'{"case_id":"example"}')).status,200);
  assert.equal(seen.at(-1).path,'/api/host/case/reverse');
  assert.equal((await request('host-case-review','PATCH',{},'{"case_id":"example","kind":"report","decision":"reject"}')).status,200);
  assert.equal(seen.at(-1).path,'/api/host/case/review');
  assert.equal(seen.at(-1).body,'{"case_id":"example","kind":"report","decision":"reject"}');
  responseStatus=404;responseBody=JSON.stringify({error:'case_not_found'});
  const missing=await request('host-case','POST',{},'{}');assert.equal(missing.status,404);
  responseStatus=409; responseBody=JSON.stringify({error:'settings_changed'});
  const conflict=await request(); assert.equal(conflict.status,409);
  assert.deepEqual(await conflict.json(),{error:'settings_changed'});
  responseStatus=500; responseBody=JSON.stringify({error:'internal details'});
  const failed=await request('settings','PATCH',{},'{}');
  assert.equal(failed.status,503); assert.deepEqual(await failed.json(),{error:'save_failed'});
  responseStatus=302; const redirect=await request(); assert.equal(redirect.status,502);
  responseStatus=200; responseBody='not JSON'; const malformed=await request(); assert.equal(malformed.status,502);
  responseBody='x'.repeat(1048577); const large=await request(); assert.equal(large.status,503);
});
test('the page limits scripts, framing and referrer leakage', async () => {
  const response=await fetch(`http://127.0.0.1:${port}/`);
  assert.equal(response.status,200); assert.equal(response.headers.get('referrer-policy'),'no-referrer');
  const csp=response.headers.get('content-security-policy');
  assert.ok(csp.includes("connect-src 'self'")); assert.ok(csp.includes('frame-ancestors https://web.telegram.org'));
  assert.equal(response.headers.get('cache-control'),'no-store');
});
