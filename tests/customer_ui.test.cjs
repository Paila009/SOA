// Lightweight frontend behavior checks; browser layout is tested separately.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'customer/web/app.js'), 'utf8').replace(/\binit\(\);\s*$/, '');
const html = fs.readFileSync(path.join(root, 'customer/web/index.html'), 'utf8');

function harness(extra={}) {
  const elements = new Map(), stored = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      value:'', hidden:true, disabled:false, textContent:'', innerHTML:'', name:'',
      attrs:{}, dataset:{}, child:{textContent:''},
      classList:{toggle(){},remove(){},add(){}},
      setAttribute(name,value){this.attrs[name]=value;},
      querySelector(){return this.child;}, focus(){}, close(){this.closed=true;},
    });
    return elements.get(id);
  }
  const context = vm.createContext({
    document:{getElementById:element},
    localStorage:{getItem:key=>stored.get(key)??null,setItem:(key,value)=>stored.set(key,value)},
    URL, setTimeout, clearTimeout, capture:null,...extra,
  });
  vm.runInContext(source,context);
  vm.runInContext(`closeSidebar=()=>{}; openModal=(title,content,eyebrow,variant)=>{capture={title,content,eyebrow,variant};}; state.config={preview:true,models:[]};`,context);
  return {context,element,stored,run:code=>vm.runInContext(code,context)};
}

test('customer preferences contain user choices, not operator setup',()=>{
  const h=harness();h.run('showSettings()');
  assert.equal(h.context.capture.title,'Make it yours.');
  assert.equal(h.context.capture.variant,'preferences');
  assert.match(h.context.capture.content,/Full answer/);
  assert.match(h.context.capture.content,/Reviewed answer/);
  assert.match(h.context.capture.content,/Look for sources/);
  assert.doesNotMatch(h.context.capture.content,/Firebase|Groq|OpenRouter|\.env|credentials|API key|Not connected|setup-steps/i);
});

test('answer style and source preferences persist and update the composer',()=>{
  const h=harness();h.run('showSettings()');
  h.element('answer-preferences').onchange({target:{name:'answer-display',value:'reviewed'}});
  assert.equal(h.run('state.strict'),true);
  assert.equal(h.stored.get('grounded:strict'),'true');
  h.element('preferences-search').onchange({target:{checked:false}});
  assert.equal(h.run('state.search'),false);
  assert.equal(h.stored.get('grounded:search'),'false');
  assert.equal(h.element('search-toggle').child.textContent,'Sources off');
  h.run('showSettings()');
  assert.match(h.context.capture.content,/value="reviewed" checked/);
  h.element('preferences-done').onclick();
  assert.equal(h.element('modal').closed,true);
});

test('unavailable chat preserves the question without opening a setup modal',async()=>{
  const h=harness();h.element('question').value='Compare solar and wind energy';
  await h.run('submitResearch()');
  assert.equal(h.element('chat-availability').hidden,false);
  assert.equal(h.element('question').value,'Compare solar and wind energy');
  assert.equal(h.context.capture,null);
  assert.match(h.element('composer-status').textContent,/live chat is not available/);
  // This also applies to signed-in customers when no model is configured.
  h.run('state.config.preview=false');h.element('chat-availability').hidden=true;
  await h.run('submitResearch()');
  assert.equal(h.element('chat-availability').hidden,false);
  assert.equal(h.context.capture,null);
  assert.doesNotMatch(h.element('composer-status').textContent,/Preview/);
});

test('preferences cannot alter an in-flight answer',()=>{
  const h=harness();h.run('state.job={jobId:"test"};showSettings()');
  assert.match(h.context.capture.content,/class="answer-preferences" disabled/);
  h.element('answer-preferences').onchange({target:{name:'answer-display',value:'reviewed'}});
  assert.equal(h.run('state.strict'),false);
});

test('customer chrome and sign-in errors do not send users to infrastructure setup',()=>{
  assert.doesNotMatch(html,/Firebase|\.env|Setup guide|Connections &amp; settings|Connections & settings/);
  assert.match(html,/Preferences/);
  assert.match(html,/Connect a model to start chatting/);
  const h=harness();
  for(const code of ['auth/unauthorized-domain','auth/operation-not-allowed','auth/configuration-not-found','auth/popup-blocked','unexpected']){
    const message=h.run(`authError({code:${JSON.stringify(code)}})`);
    assert.doesNotMatch(message,/Firebase|API key|\.env|credentials/);
    assert.ok(message.length>10);
  }
});

test('signed-in navigation never opens the login page',()=>{
  const h=harness();
  h.run('state.config={preview:false,models:[]};state.user={uid:"alice"};showAuth()');
  assert.equal(h.element('workspace').hidden,false);
  assert.equal(h.element('auth-page').hidden,true);
  h.run('showAuth(true)');
  assert.equal(h.element('workspace').hidden,true);
  assert.equal(h.element('auth-page').hidden,false);
});

test('model library keeps local and API models in the same viewer',()=>{
  const h=harness();
  h.run(`state.config={preview:false,models:[{id:'local:qwen',model:'Qwen2.5 1.5B Instruct',provider:'local',providerName:'On this computer'}],localModels:[{id:'qwen',chatId:'local:qwen',name:'Qwen2.5 1.5B Instruct',parameters:'1.54B',quantization:'Q4_K_M',license:'Apache-2.0',installed:true,downloadUrl:'https://huggingface.co/Qwen/model'}],providers:[{id:'groq',name:'Groq',enabled:true,ready:false},{id:'gemini',name:'Google Gemini',enabled:true,ready:false}]};showModels()`);
  assert.equal(h.context.capture.title,'Your model library.');
  assert.equal(h.context.capture.variant,'models');
  assert.match(h.context.capture.content,/Your installed local models still work in this same chat/);
  assert.match(h.context.capture.content,/Qwen2\.5 1\.5B Instruct/);
  assert.match(h.context.capture.content,/Ready in this chat/);
  assert.match(h.context.capture.content,/Model & download page/);
  assert.match(h.context.capture.content,/Google Gemini[\s\S]*Not connected/);
  assert.doesNotMatch(h.context.capture.content,/local research studio|127\.0\.0\.1:8766/i);
  assert.doesNotMatch(h.context.capture.content,/API key=|secret|\.env/i);
});

test('completed answers expose both verdict reasoning and searched sources',()=>{
  const h=harness();
  h.run(`state.chat={id:'chat-1',title:'Question',messages:[{id:'answer-1',role:'assistant',content:'A supported answer [1].',created:1,detail:{sources:[{id:1,title:'Source'}],review:{counts:{supported:1,unverified:0,contradicted:0}}}}]};renderChat()`);
  assert.match(h.element('chat').innerHTML,/GROUNDED CHECK/);
  assert.match(h.element('chat').innerHTML,/Why this verdict/);
  assert.match(h.element('chat').innerHTML,/View searched sources \(1\)/);
  assert.match(h.element('chat').innerHTML,/data-review-tab="sources"/);
});

test('evidence drawer names the three customer-facing review views',()=>{
  assert.match(html,/Grounded check/);
  assert.match(html,/Searched sources/);
  assert.match(html,/How it was checked/);
});

test('saved answer selection restores its own visible sources and analysis',()=>{
  const h=harness();
  h.run(`state.chat={id:'chat',title:'Research',messages:[{id:'old',role:'assistant',content:'Old answer',created:1,detail:{question:'Earlier question',sources:[{id:1,title:'Earlier source',snippet:'Earlier passage'}],review:{coverage:50,counts:{supported:1,unverified:1},claims:[{id:1,text:'Old claim',status:'supported',source:1,quote:'Earlier passage',reason:'Quoted support'}]}}},{id:'new',role:'assistant',content:'New answer',created:2,detail:{question:'New question',sources:[],review:{counts:{unverified:2},claims:[]}}}]};renderChat();selectAnswerAnalysis('old')`);
  const panel=h.element('answer-insights');
  assert.equal(panel.hidden,false);
  assert.match(panel.innerHTML,/Earlier question/);
  assert.match(panel.innerHTML,/Earlier source/);
  assert.match(panel.innerHTML,/Earlier passage/);
  assert.match(panel.innerHTML,/50%/);
  assert.doesNotMatch(panel.innerHTML,/New question/);
  assert.match(panel.innerHTML,/not a hallucination probability/);
});

test('model library offers actual personal key connection controls',()=>{
  const h=harness();h.run(`state.config={preview:false,models:[],localModels:[],providers:[]};showModels()`);
  assert.match(h.context.capture.content,/id="provider-key" type="password"/);
  assert.match(h.context.capture.content,/id="load-provider-models"/);
  assert.match(h.context.capture.content,/id="connect-provider"/);
  assert.match(h.context.capture.content,/Groq/);
  assert.match(h.context.capture.content,/Google Gemini/);
  assert.match(h.context.capture.content,/OpenRouter/);
});

test('Pages frontend initializes the runtime object and uses its protected API',async()=>{
  const {createRuntime}=require('../customer/web/browser-runtime.js');
  const records=new Map();
  const runtime=createRuntime({store:{get:async id=>records.get(id),set:async(id,value)=>records.set(id,value),delete:async id=>records.delete(id)}});
  const h=harness({GROUNDED_DEPLOYMENT:{mode:'browser',firebase:{projectId:'project',apiKey:'public-client-key'}},GroundedBrowserRuntime:runtime});
  const config=await h.run("api('/api/config')");
  assert.equal(config.firebase.projectId,'project');
  assert.equal(config.browser,true);
  await assert.rejects(h.run("api('/api/workspace')"),/verified Firebase/);
  h.run("state.user={uid:'alice',emailVerified:true}");
  const sample=await h.run("api('/api/sample',{method:'POST'})");
  assert.ok(sample.messages[1].detail.sources.length);
  const workspace=await h.run("api('/api/workspace')");
  assert.equal(workspace.chats.length,1);
  assert.equal(workspace.config.browser,true);
});
