const {test} = require('node:test');
const assert = require('node:assert/strict');
const {createRuntime,reviewAnswer} = require('../customer/web/browser-runtime.js');

const alice={uid:'alice',emailVerified:true};
const bob={uid:'bob',emailVerified:true};
const secret='personal-key-never-persist-this';
const model='text-chat-test';
function memoryStore() {
  const values=new Map();
  return {values,async get(uid){return values.has(uid)?structuredClone(values.get(uid)):null;},async set(uid,data){values.set(uid,structuredClone(data));},async delete(uid){values.delete(uid);}};
}
function sse(chunks) {
  return new Response(new ReadableStream({start(controller){for(const chunk of chunks)controller.enqueue(new TextEncoder().encode(chunk));controller.close();}}),{headers:{'Content-Type':'text/event-stream'}});
}
const groqCatalog=()=>Response.json({data:[{id:model,active:true},{id:'whisper-large-v3',active:true},{id:'guard-model',active:true},{id:'retired-chat',active:false}]});
const answer='Cricket is a team sport played between two teams of eleven players.';
function fetchMock(calls,completion=null) {
  return async(url,options={})=>{
    calls.push({url:String(url),options});
    if(String(url).endsWith('/models'))return groqCatalog();
    if(String(url).startsWith('https://en.wikipedia.org/'))return Response.json({query:{pages:{7:{pageid:7,index:1,title:'Cricket',fullurl:'https://en.wikipedia.org/wiki/Cricket',extract:answer+' Matches are played with a bat and ball.'}}}});
    if(String(url).endsWith('/chat/completions'))return completion?completion(url,options):sse(['data: {"choices":[{"delta":{"content":"Cricket is a team sport "}}]}\n','\ndata: {"choices":[{"delta":{"content":"played between two teams of eleven players. [1]"}}]}\n\ndata: [DONE]\n\n']);
    throw new Error('Unexpected request');
  };
}
async function connect(runtime,user=alice) {return runtime.connect('groq',secret,model,true,user);}
async function research(runtime,user=alice,extra={}) {
  return runtime.request('/api/research',{method:'POST',body:{question:'What is cricket?',model:`groq:${model}`,mode:'research',search:true,strict:false,documents:[],consent:true,...extra}},user);
}
async function finish(runtime,job,user=alice) {
  for(let i=0;i<100;i++){
    const snapshot=await runtime.request(`/api/jobs/${job.jobId}`,{},user);
    if(snapshot.done)return snapshot;
    await new Promise(resolve=>setTimeout(resolve,3));
  }
  throw new Error('Job failed to finish');
}

test('all workspace and provider actions require a verified Firebase account',async()=>{
  let requests=0;
  const runtime=createRuntime({store:memoryStore(),fetch:async()=>{requests++;return groqCatalog();}});
  assert.equal((await runtime.request('/api/config')).browser,true);
  await assert.rejects(()=>runtime.request('/api/workspace'),/Sign in/);
  await assert.rejects(()=>runtime.listModels('groq',secret,{uid:'unverified',emailVerified:false}),/verified Firebase/);
  assert.equal(requests,0);
});

test('model catalogs are live, chat-only, and OpenRouter is restricted to explicitly free IDs',async()=>{
  const calls=[];
  const runtime=createRuntime({store:memoryStore(),fetch:async(url,options)=>{
    calls.push({url,options});
    if(url.endsWith('/key'))return Response.json({data:{label:'Personal browser key'}});
    if(url.includes('openrouter'))return Response.json({data:[{id:'vendor/free:free',pricing:{prompt:'0',completion:'0'}},{id:'vendor/paid',pricing:{prompt:'0.01',completion:'0.01'}},{id:'vendor/not-free:free',pricing:{prompt:'1',completion:'0'}}]});
    if(url.includes('googleapis'))return Response.json({models:[{name:'models/chat-test',supportedGenerationMethods:['generateContent']},{name:'models/embed-test',supportedGenerationMethods:['embedContent']}]});
    return groqCatalog();
  }});
  assert.deepEqual((await runtime.listModels('groq',secret,alice)).map(m=>m.model),[model]);
  assert.deepEqual((await runtime.listModels('openrouter',secret,alice)).map(m=>m.model),['vendor/free:free']);
  assert.deepEqual((await runtime.listModels('gemini',secret,alice)).map(m=>m.model),['chat-test']);
  assert.equal(calls.find(c=>c.url.includes('googleapis')).options.headers['x-goog-api-key'],secret);
  assert.ok(calls.every(c=>!c.url.includes(secret)));
  await assert.rejects(()=>runtime.connect('openrouter',secret,'vendor/paid',true,alice),/free-only model list/);
});

test('provider connections require free-plan confirmation and never expose or persist keys',async()=>{
  const store=memoryStore(),runtime=createRuntime({store,fetch:fetchMock([])});
  await assert.rejects(()=>runtime.connect('groq',secret,model,false,alice),/Confirm/);
  const config=await connect(runtime);
  assert.equal(config.models[0].id,`groq:${model}`);
  assert.equal(config.providers.find(p=>p.id==='groq').personal,true);
  assert.ok(!JSON.stringify(config).includes(secret));
  await runtime.request('/api/sample',{method:'POST'},alice);
  assert.ok(!JSON.stringify([...store.values]).includes(secret));
  runtime.clearSession();
  assert.equal(runtime.publicConfig().models.length,0);
  assert.equal((await runtime.request('/api/workspace',{},alice)).config.models.length,0);
});

test('OpenRouter rejects invalid credentials before consulting its public catalog',async()=>{
  const calls=[],runtime=createRuntime({store:memoryStore(),fetch:async(url,options)=>{
    calls.push({url,options});return new Response(JSON.stringify({error:{message:secret}}),{status:401});
  }});
  await assert.rejects(()=>runtime.connect('openrouter',secret,'vendor/chat:free',true,alice),err=>/rejected this key/.test(err.message)&&!err.message.includes(secret));
  assert.equal(calls.length,1);assert.equal(calls[0].url,'https://openrouter.ai/api/v1/key');
  assert.equal(calls[0].options.headers.Authorization,`Bearer ${secret}`);
  assert.equal(runtime.publicConfig().models.length,0);
});

test('real SSE processing persists answer, exact searched sources and review history',async()=>{
  const calls=[],store=memoryStore(),runtime=createRuntime({store,fetch:fetchMock(calls)});
  await connect(runtime);
  const job=await research(runtime),done=await finish(runtime,job);
  assert.equal(done.phase,'complete');
  assert.equal(done.error,null);
  assert.match(done.text,/Cricket is a team sport/);
  const chat=await runtime.request(`/api/chats/${job.chatId}`,{},alice);
  const detail=chat.messages[1].detail;
  assert.equal(detail.sources[0].url,'https://en.wikipedia.org/wiki/Cricket');
  assert.equal(detail.review.claims[0].status,'supported');
  assert.equal(detail.review.claims[0].quote,answer);
  assert.equal(detail.review.counts.contradicted,0);
  assert.match(detail.review.limitations,/not accuracy/);
  assert.equal(new URL(calls.find(c=>c.url.includes('wikipedia')).url).searchParams.get('origin'),'*');
  const reloaded=createRuntime({store,fetch:fetchMock([])});
  assert.deepEqual((await reloaded.request(`/api/chats/${job.chatId}`,{},alice)).messages[1].detail.review,detail.review);
  assert.ok(!JSON.stringify([...store.values]).includes(secret));
  const exported=await runtime.request(`/api/chats/${job.chatId}/export`,{asBlob:true},alice);
  assert.match(await exported.text(),/SUPPORTED/);
  assert.match(await exported.text(),/en.wikipedia.org\/wiki\/Cricket/);
});

test('Firebase accounts have isolated histories and switching accounts clears provider credentials',async()=>{
  const runtime=createRuntime({store:memoryStore(),fetch:fetchMock([])});
  await connect(runtime);
  const sample=await runtime.request('/api/sample',{method:'POST'},alice);
  const workspace=await runtime.request('/api/workspace',{},bob);
  assert.equal(workspace.chats.length,0);
  assert.equal(runtime.publicConfig().models.length,0);
  await assert.rejects(()=>runtime.request(`/api/chats/${sample.id}`,{},bob),/not found/);
  assert.equal((await runtime.request('/api/workspace',{},alice)).chats.length,1);
});

test('stop aborts provider streaming and saves an explicitly incomplete draft',async()=>{
  let aborted=false;
  const runtime=createRuntime({store:memoryStore(),fetch:fetchMock([],async(url,options)=>{
    return await new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>{aborted=true;reject(Object.assign(new Error('stopped'),{name:'AbortError'}));},{once:true}));
  })});
  await connect(runtime);
  const job=await research(runtime,alice,{search:false});
  await new Promise(resolve=>setTimeout(resolve,5));
  await runtime.request(`/api/jobs/${job.jobId}/stop`,{method:'POST'},alice);
  const done=await finish(runtime,job);
  assert.equal(aborted,true);assert.equal(done.phase,'stopped');assert.equal(done.detail.cancelled,true);
  assert.match(done.detail.review.note,/not been fully reviewed/);
});

test('provider errors have no raw secret-bearing body and always finish the job',async()=>{
  const runtime=createRuntime({store:memoryStore(),fetch:fetchMock([],()=>new Response(JSON.stringify({error:{message:secret}}),{status:429}))});
  await connect(runtime);
  const done=await finish(runtime,await research(runtime));
  assert.equal(done.phase,'error');assert.equal(done.done,true);
  assert.match(done.error,/rate limit/);
  assert.ok(!JSON.stringify(done).includes(secret));
});

test('a stalled stream times out instead of leaving the chat busy forever',async()=>{
  const runtime=createRuntime({store:memoryStore(),timeoutMs:25,fetch:fetchMock([],()=>new Response(new ReadableStream({pull(){return new Promise(()=>{});}})))});
  await connect(runtime);
  const done=await finish(runtime,await research(runtime,alice,{search:false}));
  assert.equal(done.phase,'error');assert.match(done.error,/exceeded two minutes/);
});

test('Gemini uses authenticated native streaming with the chosen live catalog model',async()=>{
  let chatRequest;
  const runtime=createRuntime({store:memoryStore(),fetch:async(url,options={})=>{
    if(url.endsWith('/models'))return Response.json({models:[{name:'models/current-gemini',supportedGenerationMethods:['generateContent']}]});
    chatRequest={url,options};return sse(['data: {"candidates":[{"content":{"parts":[{"text":"Hello! How can I help?"}]}}]}\n\n']);
  }});
  await runtime.connect('gemini',secret,'current-gemini',true,alice);
  const done=await finish(runtime,await research(runtime,alice,{question:'hi',model:'gemini:current-gemini',search:true}));
  assert.equal(done.phase,'complete');assert.equal(done.detail.retrieval.status,'skipped');
  assert.match(chatRequest.url,/current-gemini:streamGenerateContent\?alt=sse$/);
  assert.equal(chatRequest.options.headers['x-goog-api-key'],secret);
  assert.ok(!chatRequest.url.includes(secret));
  assert.equal(done.detail.review.coverage,null);
});

test('document excerpts and answer reviews survive reload; workspace deletion removes them',async()=>{
  const store=memoryStore(),runtime=createRuntime({store,fetch:fetchMock([])});await connect(runtime);
  const document=await runtime.request('/api/documents',{method:'POST',headers:{'X-Filename':'Research.md'},body:new Blob([answer])},alice);
  const job=await research(runtime,alice,{search:false,documents:[document.id]});
  const done=await finish(runtime,job);
  assert.equal(done.detail.sources[0].provider,'Your document · selected excerpts');
  assert.equal(done.detail.review.claims[0].status,'supported');
  await runtime.request(`/api/documents/${document.id}`,{method:'DELETE'},alice);
  assert.equal((await runtime.request(`/api/chats/${job.chatId}`,{},alice)).messages[1].detail.sources.length,1);
  await runtime.request('/api/workspace',{method:'DELETE'},alice);
  assert.equal((await runtime.request('/api/workspace',{},alice)).chats.length,0);
  assert.equal(store.values.has('alice'),false);
});

test('passage screening avoids reversed relationships and false-support quotations',()=>{
  const source={id:1,snippet:'Fred defeated John in the tournament. It is false that Messi plays for Miami.'};
  const review=reviewAnswer('John defeated Fred in the tournament. Messi plays for Miami.',[source]);
  assert.equal(review.counts.unverified,2);
  assert.ok(review.claims.every(c=>c.source===null && c.quote===''));
  assert.equal(reviewAnswer('Hello! How can I help?',[]).coverage,null);
});

test('official local model downloads are available without falsely claiming installation',()=>{
  const config=createRuntime({store:memoryStore()}).publicConfig();
  assert.equal(config.localModels.length,3);
  assert.ok(config.localModels.every(m=>!m.installed && m.fileUrl.includes('/resolve/main/') && m.fileUrl.endsWith('?download=true')));
  assert.match(config.localModels[1].fileUrl,/microsoft\/Phi-3-mini-4k-instruct-gguf/);
});
