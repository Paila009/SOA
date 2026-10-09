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

test('localhost uses the installed-model backend even with a stale Pages flag',async()=>{
  let request;
  const h=harness({
    location:{hostname:'localhost'}, Headers, AbortController,
    GROUNDED_DEPLOYMENT:{mode:'browser'},
    GroundedBrowserRuntime:{initialize(){throw new Error('Wrong runtime');}},
    fetch:async(url,options)=>{request={url,options};return {ok:true,status:200,json:async()=>({models:[{id:'local:qwen'}],browser:false})};},
  });
  const config=await h.run("api('/api/config')");
  assert.equal(config.models[0].id,'local:qwen');
  assert.equal(h.run('browserDeployment'),false);
  assert.equal(request.url,'/api/config');
  assert.equal(request.options.cache,'no-store');
});

const installedModels=`state.config={preview:false,models:[
  {id:'local:qwen',model:'Qwen2.5 1.5B Instruct',provider:'local',providerName:'On this computer'},
  {id:'local:phi',model:'Phi-3 Mini 3.8B',provider:'local',providerName:'On this computer'},
  {id:'local:qwen3',model:'Qwen3 4B',provider:'local',providerName:'On this computer'},
  {id:'groq:api',model:'API model',provider:'groq',providerName:'Groq'}
],localModels:[{installed:true},{installed:true},{installed:true}]};`;

test('installed models remain directly selectable despite an invalid saved selection',()=>{
  const h=harness();h.stored.set('grounded:model','removed-model');
  h.run(`${installedModels}renderModels()`);
  assert.equal(h.element('model').value,'local:qwen');
  assert.equal(h.run('state.model'),'local:qwen');
  assert.equal(h.element('model').disabled,false);
  assert.match(h.element('model').innerHTML,/Downloaded models · this computer/);
  assert.match(h.element('model').innerHTML,/Connected API models/);
  assert.match(h.element('runtime-modelbar').innerHTML,/3 installed models/);
  for(const name of ['Qwen2.5 1.5B Instruct','Phi-3 Mini 3.8B','Qwen3 4B'])
    assert.ok(h.element('runtime-modelbar').innerHTML.includes(name));
  h.run("chooseModel('local:phi')");
  assert.equal(h.element('model').value,'local:phi');
  assert.equal(h.stored.get('grounded:model'),'local:phi');
  assert.match(h.element('answer-insights').innerHTML,/Phi-3 Mini 3\.8B/);
  h.run("state.job={jobId:'busy'};chooseModel('local:qwen3')");
  assert.equal(h.run('state.model'),'local:phi');
});

test('guard and source monitoring remain visible before the first question',()=>{
  const h=harness();h.run('state.config=null;renderInsights()');
  const panel=h.element('answer-insights');
  assert.equal(panel.hidden,false);
  assert.match(panel.innerHTML,/Grounded guard/);
  assert.match(panel.innerHTML,/data-monitor-search/);
  assert.match(panel.innerHTML,/data-monitor-guard/);
  assert.match(panel.innerHTML,/Searched resources/);
  assert.match(panel.innerHTML,/No search has been run yet/);
  assert.match(panel.innerHTML,/Not measured yet/);
  assert.doesNotMatch(panel.innerHTML,/\d+%|\d+ conflicting|\d+ supported/);
  h.run('setGuardEnabled(true);setSearchEnabled(false)');
  assert.equal(h.stored.get('grounded:strict'),'true');
  assert.equal(h.stored.get('grounded:search'),'false');
  assert.match(panel.innerHTML,/Reviewed only/);
  assert.match(panel.innerHTML,/Sources off/);
  h.run("state.job={jobId:'busy'};setGuardEnabled(false);setSearchEnabled(true)");
  assert.equal(h.run('state.strict'),true);
  assert.equal(h.run('state.search'),false);
});

test('live monitor uses actual retrieved passages without inventing a completed verdict',()=>{
  const h=harness();
  h.run(`${installedModels}state.model='local:qwen';state.job={jobId:'live'};state.monitorProgress={phase:'writing draft',sources:[{id:1,title:'Cricket',url:'https://en.wikipedia.org/wiki/Cricket',snippet:'Cricket is a bat-and-ball game.'}],retrieval:{note:'Actual retrieved passage'}};renderInsights()`);
  const panel=h.element('answer-insights').innerHTML;
  assert.match(panel,/In progress/);
  assert.match(panel,/writing draft/);
  assert.match(panel,/Cricket is a bat-and-ball game/);
  assert.match(panel,/https:\/\/en.wikipedia.org\/wiki\/Cricket/);
  assert.match(panel,/Actual retrieved passage/);
  assert.doesNotMatch(panel,/\d+%|Evidence matched/);
  assert.match(panel,/data-monitor-guard type="button"[^>]*disabled/);
});

test('conflicting and missing evidence keep their own reasons and source snapshots',()=>{
  const h=harness();
  h.run(`state.chat={title:'Claim test',messages:[{id:'answer',role:'assistant',created:1,detail:{sources:[{id:1,title:'Fixture source',snippet:'The fixture says 2020.'}],review:{coverage:0,method:'fixture review',counts:{contradicted:1,unverified:1},claims:[{id:1,text:'It happened in 2021.',status:'contradicted',source:1,quote:'The fixture says 2020.',reason:'The year conflicts with the cited passage.'},{id:2,text:'An unsupported claim.',status:'unverified',reason:'No retrieved passage verifies this claim.'}]}}}]};renderInsights()`);
  const panel=h.element('answer-insights').innerHTML;
  assert.match(panel,/Conflicting evidence/);
  assert.match(panel,/The year conflicts/);
  assert.match(panel,/No retrieved passage verifies/);
  assert.match(panel,/Reviewed passage · \[1\] Fixture source/);
  assert.match(panel,/unverified claim is not automatically false/);
});

test('laptop and phone styles do not hide the source and claim sections',()=>{
  const css=fs.readFileSync(path.join(root,'customer/web/style.css'),'utf8');
  assert.match(html,/id="runtime-modelbar"/);
  assert.doesNotMatch(css,/\.insights-sources\s*,\s*\.insights-claims\s*\{\s*display:\s*none/);
  assert.match(css,/\.insights-sources\s*,\s*\.insights-claims\s*\{display:block/);
  assert.match(css,/\.runtime-modelbar\{[^}]*overflow-x:auto/);
});

test('polling does not replace a manually selected saved-answer analysis',()=>{
  const h=harness();
  h.run(`state.chat={title:'History',messages:[{id:'saved',role:'assistant',created:1,detail:{question:'Earlier question',sources:[{id:1,title:'Saved source',snippet:'Saved evidence'}],review:{counts:{supported:1},claims:[{text:'Saved claim',status:'supported',source:1,quote:'Saved evidence'}]}}}]};state.job={jobId:'new-job'};state.monitorProgress={phase:'writing draft',sources:[]};selectAnswerAnalysis('saved');renderInsights()`);
  assert.match(h.element('answer-insights').innerHTML,/Earlier question/);
  assert.match(h.element('answer-insights').innerHTML,/Saved evidence/);
  assert.match(h.element('answer-insights').innerHTML,/Back to live monitor/);
  assert.doesNotMatch(h.element('answer-insights').innerHTML,/Grounded check in progress/);
  h.run('showLiveMonitor()');
  assert.match(h.element('answer-insights').innerHTML,/Grounded check in progress/);
  assert.match(h.element('answer-insights').innerHTML,/writing draft/);
});

test('submission locks models and monitoring before a job ID arrives and recovers on failure',async()=>{
  let release;
  const pendingSubmit=new Promise(resolve=>{release=resolve;});
  const h=harness({pendingSubmit});
  h.run(`${installedModels}state.user={uid:'alice'};renderModels();api=async()=>{await pendingSubmit;throw new Error('Fixture rejection');}`);
  h.element('question').value='A test question';
  const submission=h.run('submitResearch()');
  assert.equal(h.run('state.submitting'),true);
  assert.equal(h.run('state.job'),null);
  assert.equal(h.element('model').disabled,true);
  assert.match(h.element('runtime-modelbar').innerHTML,/data-use-model="local:phi" disabled/);
  assert.match(h.element('answer-insights').innerHTML,/data-monitor-guard type="button"[^>]*disabled/);
  h.run("chooseModel('local:phi');setGuardEnabled(true);setSearchEnabled(false);renderModels()");
  assert.equal(h.run('state.model'),'local:qwen');
  assert.equal(h.run('state.strict'),false);
  assert.equal(h.run('state.search'),true);
  assert.equal(h.element('model').disabled,true);
  release();await submission;
  assert.equal(h.run('state.submitting'),false);
  assert.equal(h.element('model').disabled,false);
  assert.equal(h.element('question').value,'A test question');
});

test('visible toolbar and monitor buttons are wired to actual model and guard changes',()=>{
  const h=harness({matchMedia:()=>({addEventListener(){}})});
  h.context.document.querySelectorAll=()=>[];
  h.context.document.addEventListener=()=>{};
  h.run(`${installedModels}renderModels();bindEvents()`);
  const click=(elementId,dataset,attribute)=>h.element(elementId).onclick({target:{closest:()=>({dataset,hasAttribute:name=>name===attribute})}});
  click('runtime-modelbar',{useModel:'local:qwen3'});
  assert.equal(h.run('state.model'),'local:qwen3');
  click('answer-insights',{},'data-monitor-guard');
  assert.equal(h.run('state.strict'),true);
  assert.match(h.element('answer-insights').innerHTML,/Reviewed only/);
  click('answer-insights',{},'data-monitor-search');
  assert.equal(h.run('state.search'),false);
  assert.equal(h.element('search-toggle').child.textContent,'Sources off');
});
