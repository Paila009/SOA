const $ = (id) => document.getElementById(id);
const paths = {
  plus: '<path d="M12 5v14M5 12h14"/>',
  grid: '<rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/><rect x="4" y="14" width="6" height="6" rx="1"/><rect x="14" y="14" width="6" height="6" rx="1"/>',
  book: '<path d="M12 6c-3-2-6-2-9-1v14c3-1 6-1 9 1 3-2 6-2 9-1V5c-3-1-6-1-9 1Zm0 0v14"/>',
  clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7v5l3 2"/>',
  search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4.5 4.5"/>',
  spark: '<path d="m12 2 2.7 7.3L22 12l-7.3 2.7L12 22l-2.7-7.3L2 12l7.3-2.7Z"/>',
  settings: '<path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="2.5" fill="currentColor"/><circle cx="16" cy="17" r="2.5" fill="currentColor"/>',
  chevron: '<path d="m9 6 6 6-6 6"/>',
  lock: '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 5v2"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4"/>',
  compass: '<circle cx="12" cy="12" r="9"/><path d="m16 8-2.5 5.5L8 16l2.5-5.5Z"/>',
  compare: '<path d="M4 7h15m-4-4 4 4-4 4M20 17H5m4-4-4 4 4 4"/>',
  file: '<path d="M14 3H5v18h14V8Zm0 0v5h5M8 12h8M8 16h6"/>',
  shield: '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6Zm-4 9 3 3 5-6"/>',
  globe: '<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18"/>',
  arrow: '<path d="M12 20V4m-6 6 6-6 6 6"/>',
  close: '<path d="m6 6 12 12M18 6 6 18"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  copy: '<rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V3H3v13h5"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/>'
};
const icon = (name) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.file}</svg>`;
function icons(root = document) { root.querySelectorAll('[data-icon]').forEach((el) => { el.innerHTML = icon(el.dataset.icon); }); }
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const safeLink = (url) => {
  try {
    const u = new URL(url);
    const loopback = u.protocol === 'http:' && ['localhost', '127.0.0.1', '::1'].includes(u.hostname);
    return u.protocol === 'https:' || loopback ? esc(u.href) : '';
  } catch { return ''; }
};
const state = { config:null, user:null, auth:null, firebase:null, chats:[], documents:[], selectedDocs:new Set(), chat:null, job:null, search:true, strict:false, mode:'research', model:'', review:null, reviewTab:'claims', signup:false, creatingAccount:false, epoch:0, analysisMessageId:null, sessionError:null };
const browserDeployment = globalThis.GROUNDED_DEPLOYMENT?.mode === 'browser';
let browserRuntime;
let toastTimer;
function toast(message) { $('toast').textContent=message; $('toast').hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(() => { $('toast').hidden=true; },5500); }
const timeLabel = (date) => new Date(date * 1000).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
const storage = { get(key) { try { return localStorage.getItem(key); } catch { return null; } }, set(key,value) { try { localStorage.setItem(key,value); } catch { /* Storage can be disabled without breaking chat. */ } } };
function lastChatKey() { return `grounded:customer:last:${state.user?.uid || 'preview'}`; }

async function api(path, options={}, allowRefresh=true) {
  if(browserDeployment) {
    if(!browserRuntime){browserRuntime=globalThis.GroundedBrowserRuntime;browserRuntime.initialize({firebase:globalThis.GROUNDED_DEPLOYMENT.firebase});}
    return browserRuntime.request(path,options,state.user);
  }
  const request={...options};
  const headers = new Headers(request.headers || {});
  if (state.user) headers.set('Authorization',`Bearer ${await state.user.getIdToken()}`);
  if (request.body && typeof request.body !== 'string' && !(request.body instanceof File) && !(request.body instanceof Blob)) {
    headers.set('Content-Type','application/json'); request.body=JSON.stringify(request.body);
  }
  const controller=new AbortController();
  const timeoutMs=request.timeoutMs || 30000;
  const timer=setTimeout(()=>controller.abort(),timeoutMs);
  delete request.timeoutMs;
  let response;
  try { response=await fetch(path,{...request,headers,credentials:'same-origin',signal:controller.signal}); }
  catch(err){
    if(err?.name==='AbortError')throw new Error('The server took too long to respond. Your conversation is safe; try again.');
    throw err;
  } finally { clearTimeout(timer); }
  if(response.status===401 && state.user && allowRefresh){
    await state.user.getIdToken(true);
    return api(path,options,false);
  }
  if (!response.ok) {
    let detail;
    try { detail=(await response.json()).detail; } catch { detail='The server could not be reached. Please try again.'; }
    const error=new Error(typeof detail==='string' ? detail : 'Please check the information you entered.');
    error.status=response.status;throw error;
  }
  return request.asBlob ? response.blob() : response.json();
}

function openModal(title, content, eyebrow='YOUR WORKSPACE', variant='default') {
  $('modal-title').textContent=title; $('modal-content').innerHTML=content; $('modal-eyebrow').textContent=eyebrow;
  $('modal').dataset.variant=variant; $('modal-content').scrollTop=0; $('modal').scrollTop=0;
  icons($('modal')); if (!$('modal').open) $('modal').showModal();
}
function closeSidebar() { $('sidebar').classList.remove('open'); $('sidebar-shade').hidden=true; $('menu-button').setAttribute('aria-expanded','false'); $('sidebar').inert=matchMedia('(max-width:760px)').matches; document.querySelector('.main').inert=false; }
function showAuth(force=false) { if(state.user && !state.config.preview && !force){showWorkspace();return;} closeSidebar(); $('workspace').hidden=true; $('auth-page').hidden=false; $('auth-back').hidden=!state.config.preview; $('email').focus(); }
function showWorkspace() { $('auth-page').hidden=true; $('workspace').hidden=false; }
function preventWhileRunning() { if (!state.job) return false; toast('Stop the current answer before changing conversations.'); return true; }

async function refreshWorkspace() {
  const epoch=state.epoch;
  const result=await api('/api/workspace');
  if (epoch!==state.epoch) return;
  state.chats=result.chats; state.documents=result.documents;
  if(result.config){state.config={...state.config,...result.config};renderModels();}
  state.selectedDocs=new Set([...state.selectedDocs].filter(id => state.documents.some(d => d.id===id)));
  renderHistory(); renderAttached(); $('document-count').textContent=state.documents.length;
}

function renderModels() {
  const models=state.config.models || [], saved=state.model || storage.get('grounded:model');
  $('model-count').textContent=models.filter(m=>m.provider!=='local').length+(state.config.localModels?.length || 0);
  $('model').innerHTML=models.length?models.map(m=>`<option value="${esc(m.id)}">${esc(m.model)} · ${esc(m.providerName)}</option>`).join(''):'<option value="">Connect a model in Models</option>';
  if(models.some(m=>m.id===saved))$('model').value=saved;
  state.model=$('model').value;
  $('model').disabled=!!state.job || state.config.preview || !models.length;
  $('connection-label').textContent=state.sessionError?'Workspace unavailable':state.config.preview?'Preview':models.length?'Ready to research':'Connect a model';
  $('workspace-status').title=state.sessionError || (models.length?'Choose a model and ask a question':'Open Models to connect your own API key');
}

function renderHistory() {
  const query=$('history-search').value.toLowerCase();
  const chats=state.chats.filter(c => c.title.toLowerCase().includes(query));
  $('history-list').innerHTML=chats.length ? chats.map(c => `<div class="history-item ${state.chat?.id===c.id?'selected':''}"><button data-chat="${esc(c.id)}" title="${esc(c.title)}">${esc(c.title)}</button><button class="delete-chat" data-delete-chat="${esc(c.id)}" aria-label="Delete ${esc(c.title)}">×</button></div>`).join('') : `<p class="history-empty">${query?'No matching conversations.':'Your ideas will find a home here.<br>Start a question or explore the sample.'}</p>`;
}

function renderAttached() {
  $('attached-docs').innerHTML=[...state.selectedDocs].map(id => { const d=state.documents.find(d => d.id===id); return d?`<div class="doc-chip"><span>${esc(d.name)}</span><button data-unattach="${id}" aria-label="Stop using ${esc(d.name)}">×</button></div>`:''; }).join('');
}

function inlineText(text, messageId) {
  let html=esc(text).replace(/\*\*([^*\n]+)\*\*/g,'<strong>$1</strong>');
  if (messageId) html=html.replace(/\[(\d{1,2})\]/g,(_,n)=>`<button class="citation" data-cite="${n}" data-message="${esc(messageId)}" aria-label="View source ${n}">${n}</button>`);
  return html;
}

function richText(text,messageId) {
  const rows=text.split('\n'); let html='', bullets=[];
  const flush=()=>{if(bullets.length){html+='<ul>'+bullets.map(b=>`<li>${inlineText(b,messageId)}</li>`).join('')+'</ul>';bullets=[];}};
  for (const row of rows) {
    if (/^\s*[-*•]\s/.test(row)) { bullets.push(row.replace(/^\s*[-*•]\s+/,'')); continue; }
    flush();
    if (!row.trim()) continue;
    if (/^#{1,4}\s/.test(row)) html+=`<h3 class="answer-heading">${inlineText(row.replace(/^#{1,4}\s+/,''),messageId)}</h3>`;
    else html+=`<p>${inlineText(row,messageId)}</p>`;
  }
  flush(); return html;
}

function renderInsights(message=null) {
  const panel=$('answer-insights'), scroll=$('content-scroll');
  if(!panel || !scroll)return;
  const answers=(state.chat?.messages || []).filter(m=>m.role==='assistant');
  const selected=message || answers.find(m=>m.id===state.analysisMessageId) || answers[answers.length-1];
  panel.hidden=!selected;
  scroll.classList.toggle('has-insights',!!selected);
  if(!selected){panel.innerHTML='';state.analysisMessageId=null;return;}
  state.analysisMessageId=selected.id;
  const d=selected.detail || {}, review=d.review || {}, counts=review.counts || {}, sources=d.sources || [], claims=review.claims || [];
  const supported=counts.supported || 0, contradicted=counts.contradicted || 0, unverified=counts.unverified || 0, inspect=contradicted+unverified;
  const coverage=Number.isFinite(review.coverage)?Math.max(0,Math.min(100,review.coverage)):null;
  const conversational=claims.length>0&&supported+contradicted+unverified===0;
  const verdict=d.cancelled?'Stopped':d.error?'Review incomplete':!claims.length?'Not reviewed':conversational?'Conversational':contradicted?'Conflicting evidence':!sources.length&&supported===0?'No evidence':unverified?'Needs evidence':'Evidence matched';
  const tone=d.cancelled||d.error||contradicted?'conflict':unverified||!claims.length||(!conversational&&!sources.length&&supported===0)?'attention':'matched';
  const guard=d.cancelled?'Generation stopped':d.error?'Review did not finish':d.strict?(inspect?'Weak claims filtered':'Reviewed answer'):'Full answer + review';
  const risk=conversational?'No factual claims needed source verification in this answer.':contradicted?`${contradicted} claim${contradicted===1?'':'s'} conflict with the reviewed passages.`:unverified?`${unverified} claim${unverified===1?' lacks':'s lack'} cited support in this review.`:claims.length?'No conflicting or unverified claims were reported by this review.':'This answer has no completed claim review yet.';
  const claimOrder={contradicted:0,unverified:1,supported:2,not_factual:3};
  const visibleClaims=[...claims].sort((a,b)=>(claimOrder[a.status]??4)-(claimOrder[b.status]??4)).slice(0,3);
  panel.innerHTML=`<header class="insights-heading"><div><span class="eyebrow">BEHIND THE ANSWER</span><h2>Answer insights.</h2></div><span class="insights-verdict ${tone}">${esc(verdict)}</span></header>
    <div class="insights-selected"><span>Selected answer · ${esc(timeLabel(selected.created))}</span><p>${esc(d.question || state.chat?.title || 'Saved answer')}</p>${d.sample?'<span class="sample-badge">Illustrative sample · not a measured result</span>':''}</div>
    <section class="insights-signal ${tone}"><span class="insights-signal-icon">${icon('shield')}</span><div><strong>Evidence risk signal</strong><p>${esc(risk)}</p></div></section>
    <div class="insights-metrics"><div><span>Evidence coverage</span><strong>${coverage===null?'N/A':`${Math.round(coverage)}%`}</strong><small>Cited factual claims · not accuracy</small></div><div><span>Claims to inspect</span><strong>${inspect}</strong><small>${contradicted} conflicting · ${unverified} unverified</small></div><div><span>Guard action</span><strong class="insights-metric-text">${esc(guard)}</strong><small>${d.strict?'Reviewed-only display':'Original answer visible'}</small></div><div><span>Response time</span><strong class="insights-metric-text">${Number.isFinite(d.elapsed)?`${d.elapsed.toFixed(1)} s`:'Not recorded'}</strong><small>Retrieval + generation + review</small></div></div>
    <div class="insights-controls"><button class="review-button" data-review="${esc(selected.id)}" data-review-tab="claims">View full analysis ${icon('chevron')}</button><button class="source-button" data-review="${esc(selected.id)}" data-review-tab="sources">Searched sources (${sources.length})</button><button class="source-button" data-review="${esc(selected.id)}" data-review-tab="draft">Review setup</button></div>
    <section class="insights-section insights-sources"><div class="insights-section-title"><h3>${icon('file')} Searched sources</h3><button class="text-button" data-review="${esc(selected.id)}" data-review-tab="sources">${sources.length} source${sources.length===1?'':'s'} ↗</button></div><p class="insights-section-note">${esc(d.retrieval?.note || 'Exact passages available to this answer.')}</p>${sources.length?sources.slice(0,4).map(s=>`<article class="insights-source"><button data-review="${esc(selected.id)}" data-review-tab="sources">[${esc(s.id)}] ${esc(s.title)} ↗</button><small>${esc(s.provider || 'Retrieved passage')}</small><p>${esc(String(s.snippet || '').slice(0,220))}${String(s.snippet || '').length>220?'…':''}</p></article>`).join(''):'<p class="insights-empty">No source passages were available. A missing source does not mean the answer is false.</p>'}</section>
    <section class="insights-section insights-claims"><div class="insights-section-title"><h3>${icon('shield')} Claim review</h3><span>${claims.length} claims</span></div>${visibleClaims.length?visibleClaims.map(c=>{const status=['supported','unverified','contradicted','not_factual'].includes(c.status)?c.status:'unverified';const source=sources.find(s=>String(s.id)===String(c.source));return `<article class="insights-claim ${status}"><span>${esc(status.replace('_',' '))}</span><p>${esc(c.text)}</p><small>${esc(c.reason || 'No reason recorded.')}</small>${c.quote&&source?`<details><summary>Reviewed passage · [${esc(source.id)}] ${esc(source.title)}</summary><blockquote>${esc(c.quote)}</blockquote><button class="text-button" data-review="${esc(selected.id)}" data-review-tab="sources">View source ↗</button></details>`:''}</article>`;}).join(''):'<p class="insights-empty">Claim-level reasons will appear when review finishes.</p>'}${claims.length>3?`<button class="text-button insights-all-claims" data-review="${esc(selected.id)}" data-review-tab="claims">Inspect all ${claims.length} claims ↗</button>`:''}</section>
    <footer class="insights-limit">An unverified claim is not automatically false. Evidence coverage is not a hallucination probability. Select “Why this verdict” on any older answer to see its saved review.</footer>`;
}

function selectAnswerAnalysis(messageId) {
  const message=state.chat?.messages.find(m=>m.id===messageId&&m.role==='assistant');
  if(!message)return;
  state.analysisMessageId=messageId;
  renderInsights(message);
}

function renderChat(scrollToEnd=false) {
  const chat=state.chat, isChat=!!chat;
  $('welcome').hidden=isChat; $('chat').hidden=!isChat; $('export-button').hidden=!isChat;
  $('page-title').textContent=chat?.title || 'New research';
  if (!chat) { $('chat').innerHTML=''; renderHistory(); renderInsights(); return; }
  let html=`<div class="chat-meta">${icon('lock')} Saved in your workspace ${chat.sample?'<span class="sample-badge">Illustrative sample</span>':''}</div>`;
  html+=chat.messages.map(m => {
    if (m.role==='user') return `<article class="message user"><div class="user-bubble">${esc(m.content)}</div></article>`;
    const d=m.detail || {}, counts=d.review?.counts || {}, sourceCount=d.sources?.length || 0;
    const inspect=(counts.unverified || 0)+(counts.contradicted || 0);
    const verdict=d.cancelled?'Stopped before review':d.error?'Review incomplete':sourceCount===0?'No sources retrieved':inspect>0?'Some claims need review':'Claims matched the available evidence';
    return `<article class="message assistant" id="message-${esc(m.id)}"><div class="assistant-header"><img src="./assets/logo.svg" alt=""> Grounded ${d.sample?'<span class="sample-badge">Sample answer</span>':''}<small>${esc(timeLabel(m.created))}</small></div><div class="message-content">${richText(m.content,m.id)}</div><section class="evidence-strip" aria-label="Evidence review for this answer"><div class="evidence-strip-copy"><span class="evidence-kicker">${icon('shield')} GROUNDED CHECK</span><strong>${esc(verdict)}</strong><small>${sourceCount} searched source${sourceCount===1?'':'s'} · ${counts.supported || 0} supported · ${inspect} to inspect</small></div><div class="evidence-actions"><button class="review-button" data-review="${esc(m.id)}" data-review-tab="claims">Why this verdict <span>↗</span></button><button class="source-button" data-review="${esc(m.id)}" data-review-tab="sources">View searched sources (${sourceCount})</button><button class="icon-button" data-copy="${esc(m.id)}" aria-label="Copy this answer">${icon('copy')}</button></div></section></article>`;
  }).join('');
  if (state.job) html+=`<article id="pending-answer" class="message assistant"><div class="assistant-header"><img src="./assets/logo.svg" alt=""> Grounded</div><div class="phase-line" role="status"><span class="phase-dot"></span><span id="job-phase">Getting started…</span></div><div id="draft-notice" class="draft-notice">Draft in progress · evidence review follows generation</div><div id="job-text" class="message-content"></div></article>`;
  $('chat').innerHTML=html; renderHistory(); if(scrollToEnd) $('content-scroll').scrollTop=$('content-scroll').scrollHeight;
  renderInsights();
}

async function loadChat(id) {
  if(preventWhileRunning()) return;
  const epoch=state.epoch; const chat=await api(`/api/chats/${encodeURIComponent(id)}`);
  if(epoch!==state.epoch) return;
  state.chat=chat; storage.set(lastChatKey(),id); renderChat(true); closeSidebar();
}

function newChat(focus=true) {
  if(preventWhileRunning()) return;
  $('chat-availability').hidden=true;
  state.chat=null; storage.set(lastChatKey(),''); renderChat(); closeSidebar(); if(focus)$('question').focus(); $('content-scroll').scrollTop=0;
}

async function loadSample() {
  if(preventWhileRunning()) return;
  $('chat-availability').hidden=true;
  try { state.chat=await api('/api/sample',{method:'POST'}); await refreshWorkspace(); renderChat(true); storage.set(lastChatKey(),state.chat.id); closeSidebar(); }
  catch(err){toast(err.message);}
}

function showSettings() {
  closeSidebar();
  openModal('Make it yours.', `<p class="preferences-intro">A few small choices for the way you research.</p>
    <fieldset id="answer-preferences" class="answer-preferences" ${state.job?'disabled':''}><legend>How would you like your answers?</legend>
      <label class="answer-option"><input type="radio" name="answer-display" value="full" ${state.strict?'':'checked'}><span class="preference-icon">${icon('spark')}</span><span class="preference-copy"><strong>Full answer</strong><small>Read as it’s written. Explore the evidence review afterward.</small></span><span class="selection-indicator" aria-hidden="true"></span></label>
      <label class="answer-option"><input type="radio" name="answer-display" value="reviewed" ${state.strict?'checked':''}><span class="preference-icon">${icon('shield')}</span><span class="preference-copy"><strong>Reviewed answer</strong><small>Wait for the review, then filter claims with weak evidence.</small></span><span class="selection-indicator" aria-hidden="true"></span></label>
    </fieldset>
    <label class="preference-search"><span class="preference-icon">${icon('globe')}</span><span class="preference-copy"><strong>Look for sources</strong><small>Search for background reading with your question.</small></span><input id="preferences-search" type="checkbox" role="switch" ${state.search?'checked':''} ${state.job?'disabled':''}></label>
    <div class="preferences-footer"><span id="preferences-feedback" role="status">${state.job?'Preferences unlock when this answer finishes.':'Saved in this browser.'}</span><button id="preferences-done" class="primary-button" type="button">Done <span>✓</span></button></div>`, 'PREFERENCES', 'preferences');
  $('answer-preferences').onchange=(e)=>{if(e.target.name!=='answer-display'||state.job)return;state.strict=e.target.value==='reviewed';storage.set('grounded:strict',String(state.strict));$('preferences-feedback').textContent='Preference saved.';};
  $('preferences-search').onchange=(e)=>{if(state.job)return;setSearchEnabled(e.target.checked);$('preferences-feedback').textContent='Preference saved.';};
  $('preferences-done').onclick=()=>$('modal').close();
}

function setSearchEnabled(enabled) {
  state.search=enabled; storage.set('grounded:search',String(enabled));
  $('search-toggle').classList.toggle('selected',enabled); $('search-toggle').setAttribute('aria-pressed',String(enabled));
  $('search-toggle').querySelector('span:last-child').textContent=enabled?'Sources on':'Sources off';
}

function showModels() {
  closeSidebar();
  const usable=state.config.models || [], apiModels=usable.filter(m=>m.provider!=='local'), local=state.config.localModels || [], providers=state.config.providers || [];
  const apiCards=apiModels.length?apiModels.map(m=>`<article class="model-card ready"><span class="model-status">Ready for chat</span><h3>${esc(m.model)}</h3><p>${esc(m.providerName)} · ${browserDeployment?'your key stays in this tab':'key held in server memory'}</p><button class="text-button" data-use-model="${esc(m.id)}">Use this model →</button></article>`).join(''):`<p class="empty-state">Connect your own provider key below. ${browserDeployment?'No shared API keys are embedded in this website.':'Your installed local models still work in this same chat.'}</p>`;
  const files={ 'qwen2.5-1.5b':'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true', 'phi-3-mini':'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf?download=true', 'qwen3-4b':'https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf?download=true' };
  const localCards=local.map(m=>`<article class="model-card ${m.installed?'installed':'empty'}"><span class="model-status">${m.installed?'Ready in this chat':'Download for local use'}</span><h3>${esc(m.name)}</h3><p>${esc(m.parameters)} parameters · ${esc(m.quantization)} · ${esc(m.license)}</p>${m.installed?`<button class="text-button" data-use-model="${esc(m.chatId)}">Use this model →</button>`:''}${safeLink(m.fileUrl || files[m.id])?`<a class="model-download" href="${safeLink(m.fileUrl || files[m.id])}" target="_blank" rel="noopener noreferrer">${icon('download')} Download model weights</a>`:''}${safeLink(m.downloadUrl)?`<a class="model-download" href="${safeLink(m.downloadUrl)}" target="_blank" rel="noopener noreferrer">Model & download page ↗</a>`:''}</article>`).join('');
  const providerLinks={groq:'https://console.groq.com/keys',gemini:'https://aistudio.google.com/apikey',openrouter:'https://openrouter.ai/settings/keys'};
  const providerCards=providers.map(p=>`<div class="provider-row"><span><strong>${esc(p.name)}</strong><small><a href="${providerLinks[p.id]}" target="_blank" rel="noopener noreferrer">Get your own key ↗</a></small></span><span class="provider-state ${p.ready?'ready':''}">${p.ready?'Connected':'Not connected'}</span>${p.personal || p.connection==='personal'?`<button class="text-button" data-disconnect-provider="${esc(p.id)}">Disconnect</button>`:''}</div>`).join('');
  openModal('Your model library.', `<p class="model-intro">Local and API models, one research workspace. Choose the model that fits your question.</p><section class="model-group"><div class="model-group-heading"><h3>API models</h3><span>${apiModels.length} connected</span></div><div class="model-grid">${apiCards}</div><div class="provider-list">${providerCards}</div><form id="provider-form" class="provider-connect"><h3>Connect your API key</h3><p>${browserDeployment?'Your key goes directly to the selected provider and stays only in memory in this tab. Closing or refreshing the tab disconnects it. Use your own key on a trusted device.':'Your key is held only in server memory for your signed-in account, never saved in chats. Restarting the server disconnects it.'}</p><label for="provider-choice">Provider</label><select id="provider-choice"><option value="groq">Groq</option><option value="gemini">Google Gemini</option><option value="openrouter">OpenRouter · :free models</option></select><label for="provider-key">Your API key</label><input id="provider-key" type="password" autocomplete="off" spellcheck="false" required placeholder="Paste your personal provider key"><div class="button-row"><button id="load-provider-models" class="secondary-button" type="button">Find available models</button></div><label for="provider-model">Available model</label><select id="provider-model" required disabled><option value="">Load the provider’s model list first</option></select><label class="free-confirm"><input id="free-confirm" type="checkbox" required> I use this provider’s free plan with billing disabled. Quotas and availability still apply.</label><p class="privacy-small">OpenRouter is restricted to explicit :free model IDs. This app cannot verify your Groq/Gemini billing plan; do not connect a billed project if you want free-only usage.</p><p id="provider-feedback" role="status"></p><button id="connect-provider" class="primary-button" type="submit" disabled>Connect and use model →</button></form></section><section class="model-group"><div class="model-group-heading"><h3>Local models</h3><span>${local.filter(m=>m.installed).length}/${local.length} ready here</span></div><div class="model-grid">${localCards}</div><p class="model-footnote">Downloading a file does not run it in your browser. Install the local app and its runtime, then open the same workspace on localhost. Your existing installed models are available there.</p><div class="button-row"><a class="secondary-button" href="https://github.com/Paila009/SOA#run-the-customer-workspace" target="_blank" rel="noopener noreferrer">Local setup instructions ↗</a><a class="text-button" href="https://github.com/Paila009/SOA/archive/refs/heads/main.zip">Download local app</a></div></section>`, 'MODELS & RUNTIMES', 'models');
  const resetModels=()=>{$('provider-model').innerHTML='<option value="">Load the provider’s model list first</option>';$('provider-model').disabled=true;$('connect-provider').disabled=true;};
  $('provider-choice').onchange=resetModels;$('provider-key').oninput=resetModels;
  $('load-provider-models').onclick=async()=>{
    if(!$('provider-key').value.trim()){toast('Enter your own provider key first.');return;}
    $('load-provider-models').disabled=true;$('provider-feedback').textContent='Checking the provider’s current model list…';
    try{const result=await api('/api/providers/models',{method:'POST',body:{provider:$('provider-choice').value,key:$('provider-key').value.trim()}});$('provider-model').innerHTML=result.models.map(m=>`<option value="${esc(m.id)}">${esc(m.name || m.model || m.id)}</option>`).join('');$('provider-model').disabled=!result.models.length;$('connect-provider').disabled=!result.models.length;$('provider-feedback').textContent=result.models.length?'Choose a model, then connect.':'No compatible models were returned.';}catch(err){resetModels();$('provider-feedback').textContent=err.message;}finally{$('load-provider-models').disabled=false;}
  };
  $('provider-form').onsubmit=async(e)=>{
    e.preventDefault();$('connect-provider').disabled=true;
    try{const cfg=await api('/api/providers/connect',{method:'POST',body:{provider:$('provider-choice').value,key:$('provider-key').value.trim(),model:$('provider-model').value,freeConfirmed:$('free-confirm').checked}});state.config={...state.config,...cfg};state.model=`${$('provider-choice').value}:${$('provider-model').value}`;$('provider-key').value='';renderModels();$('chat-availability').hidden=true;$('modal').close();toast('Model connected. You can chat now.');$('question').focus();}catch(err){$('provider-feedback').textContent=err.message;$('connect-provider').disabled=false;}
  };
  $('modal-content').onclick=async(e)=>{
    const use=e.target.closest('[data-use-model]'),disconnect=e.target.closest('[data-disconnect-provider]');
    if(use){if(state.job){toast('Stop the current answer before switching models.');return;}state.model=use.dataset.useModel;renderModels();storage.set('grounded:model',state.model);$('modal').close();$('question').focus();}
    if(disconnect){try{state.config={...state.config,...await api('/api/providers/disconnect',{method:'POST',body:{provider:disconnect.dataset.disconnectProvider}})};renderModels();showModels();}catch(err){toast(err.message);}}
  };
}

function showChatUnavailable() {
  $('chat-availability').hidden=false;
  $('composer-status').textContent=state.config.preview?'Preview · live chat is not available yet.':'Live chat is temporarily unavailable.';
}

function showLibrary() {
  closeSidebar();
  openModal('Bring your own context.', `<p>Add text from papers, notes, or a report. Select up to three documents to use in your next answer. Selected passages stay local with a local model and are sent to the provider only when you select an API model.</p><label class="upload-zone" for="file-upload">+ Add a document<small>UTF-8 .txt or .md · up to 40,000 characters<br>For PDFs, export the relevant text first.</small></label><input id="file-upload" type="file" accept=".txt,.md,text/plain,text/markdown" hidden><div id="document-list">${state.documents.length?state.documents.map(d=>`<div class="document-row"><label><input type="checkbox" data-doc="${d.id}" ${state.selectedDocs.has(d.id)?'checked':''}><span>${esc(d.name)}<small>${d.characters.toLocaleString()} characters</small></span></label><button data-delete-doc="${d.id}" aria-label="Delete ${esc(d.name)}">Remove</button></div>`).join(''):'<div class="empty-state">A little context goes a long way.<br>Your document library is empty.</div>'}</div><p>Removing a document from the library does not remove excerpts already saved in answer reviews. Delete those conversations separately.</p><div class="button-row"><button class="primary-button" id="library-done">Use selected documents <span>↗</span></button></div>`, 'YOUR DOCUMENT LIBRARY');
  $('library-done').onclick=()=>{$('modal').close();$('question').focus();};
  $('file-upload').onchange=async(e)=>{
    const file=e.target.files[0];if(!file)return;
    if(file.size>2*1024*1024){toast('Choose a smaller text file (maximum 2 MB).');return;}
    e.target.disabled=true;
    try { const d=await api('/api/documents',{method:'POST',headers:{'X-Filename':encodeURIComponent(file.name)},body:file});if(state.selectedDocs.size<3)state.selectedDocs.add(d.id);await refreshWorkspace();showLibrary();toast('Document added to your private library.'); }
    catch(err){toast(err.message);e.target.disabled=false;}
  };
  $('document-list').onchange=e=>{ const id=e.target.dataset.doc;if(!id)return;if(e.target.checked&&state.selectedDocs.size>=3){e.target.checked=false;toast('Use up to three documents per answer.');return;}e.target.checked?state.selectedDocs.add(id):state.selectedDocs.delete(id);renderAttached(); };
  $('document-list').onclick=async(e)=>{const button=e.target.closest('[data-delete-doc]');if(!button)return;if(!confirm('Remove this document from your library? Saved answer excerpts will remain.'))return;try{await api(`/api/documents/${button.dataset.deleteDoc}`,{method:'DELETE'});await refreshWorkspace();showLibrary();}catch(err){toast(err.message);}};
}

function showAccount() {
  if(state.config.preview){showAuth();return;}
  closeSidebar();
  openModal('Your workspace, your choice.', `<p>Signed in as <strong>${esc(state.user?.email || 'Researcher')}</strong>.</p><p>Your conversations and document library belong to your account. Export anything you want to keep before removing it.</p><div class="button-row"><button id="sign-out" class="secondary-button">Sign out</button><button id="clear-data" class="danger-button">Delete all workspace data</button></div><p>Deleting workspace data keeps your login account. Usage counts remain for up to 24 hours to protect request limits.</p>`, 'ACCOUNT & PRIVACY');
  $('sign-out').onclick=async()=>{if(preventWhileRunning())return;$('modal').close();await state.firebase.signOut(state.auth);};
  $('clear-data').onclick=async()=>{
    if(preventWhileRunning())return;
    if(!confirm('Permanently delete all your conversations, saved reviews, and uploaded documents? This cannot be undone.'))return;
    try{await api('/api/workspace',{method:'DELETE'});state.chat=null;state.selectedDocs.clear();await refreshWorkspace();renderChat();$('modal').close();toast('Your workspace data was deleted.');}catch(err){toast(err.message);}
  };
}

function openReview(messageId, tab='claims', sourceId=null) {
  const message=state.chat?.messages.find(m=>m.id===messageId);
  if(!message){toast('This answer review is not available in the current conversation.');return;}
  state.review=message;state.reviewTab=tab;
  selectAnswerAnalysis(messageId);
  $('review-question').textContent=message.detail?.question || state.chat.title;
  renderReview();if(!$('review-dialog').open)$('review-dialog').showModal();
  if(sourceId){const target=$(`source-${sourceId}`);if(target)target.scrollIntoView({block:'start'});else toast('That citation has no retrieved source. Treat it as unverified.');}
}

function renderReview() {
  const message=state.review;if(!message)return;
  const d=message.detail || {}, review=d.review || {}, counts=review.counts || {};
  $('review-tabs').querySelectorAll('[data-tab]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.tab===state.reviewTab)));
  let html='';
  if(d.sample)html+='<div class="notice">Illustrative sample. This answer and its review were hand-authored to demonstrate the interface, not generated by a model.</div>';
  if(state.reviewTab==='claims') {
    html+=`<div class="review-summary"><div><strong>${counts.supported || 0}</strong><small>Supported</small></div><div class="unverified"><strong>${counts.unverified || 0}</strong><small>Unverified</small></div><div class="contradicted"><strong>${counts.contradicted || 0}</strong><small>Contradicted</small></div></div><p class="review-note">${esc(review.note || 'No completed review is available.')} ${review.coverage!==null&&review.coverage!==undefined?`${review.coverage}% of reviewed factual claims have cited support. This is coverage, not accuracy.`:''}</p>`;
    html+=(review.claims || []).map(c=>`<article class="review-card ${esc(c.status)}"><span class="claim-status">${esc(c.status.replace('_',' '))} · claim ${c.id}</span><p>${esc(c.text)}</p><p class="claim-reason">${esc(c.reason)}</p>${c.quote?`<details><summary>Read the supporting passage · [${c.source}]</summary><blockquote>${esc(c.quote)}</blockquote><button class="text-button passage-source" data-source-tab="${c.source}">Open source ↗</button></details>`:''}</article>`).join('');
    if(!review.claims?.length)html+='<p class="empty-state">No claims have been reviewed for this answer.</p>';
  } else if(state.reviewTab==='sources') {
    html+=`<p class="review-note">${esc(d.retrieval?.note || 'The exact passages available for this answer are shown below.')} A retrieved page is not automatically an authoritative source.</p>`;
    html+=(d.sources || []).map(s=>{const url=safeLink(s.url);return `<article class="source-card" id="source-${s.id}"><small>${esc(s.provider || 'Retrieved passage')}</small><h3>${url?`<a href="${url}" target="_blank" rel="noopener noreferrer">[${s.id}] ${esc(s.title)} ↗</a>`:`[${s.id}] ${esc(s.title)}`}</h3><p>${esc(s.snippet)}</p></article>`;}).join('') || '<p class="empty-state">No source passages were available.<br>Missing evidence is not proof of a false answer.</p>';
  } else {
    html+=`<div class="detail-grid"><div><small>Answer model</small>${esc(d.model || 'Not recorded')}</div><div><small>Review model</small>${esc(d.reviewModel || (d.sample?'Hand-authored sample':'Not recorded'))}</div><div><small>Saved</small>${esc(timeLabel(message.created))}</div><div><small>End-to-end time</small>${d.elapsed!==undefined&&d.elapsed!==null?`${esc(d.elapsed)} seconds`:'Not measured'}</div><div><small>Review method</small>${esc(review.method || 'Not available')}</div><div><small>Display policy</small>${d.strict?'Reviewed claims only; original draft preserved below.':'Full answer, with evidence review attached.'}</div></div><p class="review-note">${esc(review.limitations || 'Reviews can make mistakes. Inspect important claims and source quality yourself.')}</p>${d.original?`<h3>Original unfiltered draft</h3><div class="original-draft">${esc(d.original)}</div>`:''}`;
  }
  $('review-content').innerHTML=html;$('review-content').scrollTop=0;
}

function setBusy(busy) {
  $('send-button').hidden=busy;$('stop-button').hidden=!busy;
  for(const id of ['question','model','mode','search-toggle'])$(id).disabled=busy;
  $('model').disabled=busy||state.config.preview||!state.config.models.length;
  $('composer-status').textContent=busy?'Working through your question…':state.config.preview?'Preview · live chat is not available yet.':'Thoughtful answers. Transparent evidence.';
}

async function submitResearch() {
  if(state.job)return;
  const question=$('question').value.trim();if(!question){$('question').focus();return;}
  if(state.sessionError){toast(state.sessionError);return;}
  if(state.config.preview || !$('model').value){showChatUnavailable();return;}
  const model=$('model').value;
  const selected=state.config.models.find(m=>m.id===model);
  if(selected?.provider!=='local' && storage.get(`grounded:consent:${state.user.uid}:${model}`)!=='yes'){
    openModal('Before we follow that question.', `<p>Your question, recent conversation, and selected evidence will be sent to <strong>${esc(selected?.providerName || 'your model provider')}</strong> for generation and review. Public searches also send your search terms to Wikipedia.</p><p>Do not include confidential or personal material unless you are authorized to share it with these services. This application does not guarantee provider retention or confidentiality.</p><div class="button-row"><button id="consent-send" class="primary-button">I understand · send question <span>↗</span></button><button id="consent-cancel" class="secondary-button">Cancel</button></div>`, 'A NOTE ON YOUR DATA');
    $('consent-cancel').onclick=()=>$('modal').close();
    $('consent-send').onclick=()=>{storage.set(`grounded:consent:${state.user.uid}:${model}`,'yes');$('modal').close();submitResearch();};return;
  }
  setBusy(true);$('send-button').disabled=true;
  try{
    const result=await api('/api/research',{method:'POST',body:{chatId:state.chat?.sample?null:(state.chat?.id || null),question,model,mode:$('mode').value,search:state.search,strict:state.strict,documents:[...state.selectedDocs],consent:selected?.provider!=='local'}});
    state.job=result;state.chat=await api(`/api/chats/${result.chatId}`);$('question').value='';storage.set(lastChatKey(),result.chatId);renderChat(true);await refreshWorkspace();await pollJob(result);
  }catch(err){
    if(err.status===401){state.sessionError='Your session could not be verified. Sign in again using the account menu.';renderModels();}
    toast(err.message);
  }finally{state.job=null;setBusy(false);$('send-button').disabled=false;}
}

async function pollJob(job) {
  let failures=0;
  const epoch=state.epoch;
  const deadline=Date.now()+390000;
  while(state.job?.jobId===job.jobId && epoch===state.epoch){
    if(Date.now()>deadline){
      try{await api(`/api/jobs/${job.jobId}/stop`,{method:'POST',timeoutMs:8000});}catch{/* The request may already have ended. */}
      throw new Error('This answer exceeded the six-minute safety limit and was stopped. Try a shorter question or another model.');
    }
    try{
      const result=await api(`/api/jobs/${job.jobId}`,{timeoutMs:10000});failures=0;
      if(epoch!==state.epoch)return;
      const scroll=$('content-scroll'), nearBottom=scroll.scrollHeight-scroll.scrollTop-scroll.clientHeight<130;
      const seconds=Math.max(0,Math.round(Date.now()/1000-(result.createdAt || Date.now()/1000)));
      if($('job-phase'))$('job-phase').textContent=result.phase.charAt(0).toUpperCase()+result.phase.slice(1)+`… ${seconds}s`;
      if($('job-text'))$('job-text').innerHTML=richText(result.text);
      if($('draft-notice'))$('draft-notice').textContent=state.strict?'Draft hidden until its evidence review completes.':'Draft in progress · evidence review follows generation';
      if(nearBottom)scroll.scrollTop=scroll.scrollHeight;
      if(result.done){
        state.job=null;state.chat=await api(`/api/chats/${job.chatId}`);state.analysisMessageId=null;renderChat(true);await refreshWorkspace();
        if(result.error){const error=document.createElement('div');error.className='request-error';error.textContent=result.error;$('chat').append(error);toast('The request did not finish. See the message in your conversation.');}
        return;
      }
    }catch(err){failures++;if(failures>=4){try{await api(`/api/jobs/${job.jobId}/stop`,{method:'POST'});}catch{/* Connection may be unavailable. */}throw new Error('Connection lost. A stop was requested; server-side time limits still apply. Refresh to recover your saved conversation.');}}
    await new Promise(resolve=>setTimeout(resolve,550));
  }
}

async function startWorkspace() {
  showWorkspace();$('account-name').textContent=state.user?.displayName || state.user?.email || 'Preview workspace';
  $('avatar').textContent=(state.user?.displayName || state.user?.email || 'G')[0].toUpperCase();
  $('account-plan').textContent=state.config.preview?'Explore at your own pace':'Personal research workspace';
  await refreshWorkspace();
  state.sessionError=null;renderModels();
  const last=storage.get(lastChatKey());
  if(last&&state.chats.some(c=>c.id===last))await loadChat(last);else newChat(false);
}

function authError(err){
  const messages={'auth/invalid-credential':'The email or password was not accepted. Please try again.','auth/email-already-in-use':'An account already uses that email. Sign in or reset your password.','auth/popup-closed-by-user':'Sign-in was closed before it finished.','auth/popup-blocked':'Your browser blocked the Google sign-in window. Allow pop-ups for this site and try again.','auth/cancelled-popup-request':'Another sign-in window is already open. Complete or close it, then try again.','auth/configuration-not-found':'Sign-in isn’t ready yet. Please contact the workspace owner.','auth/unauthorized-domain':'Sign-in is unavailable at this address. Please contact the workspace owner.','auth/operation-not-allowed':'This sign-in option isn’t available yet. Please try another option.','auth/too-many-requests':'Too many sign-in attempts. Please wait before trying again.','auth/network-request-failed':'The sign-in service could not be reached. Please try again.'};
  return messages[err.code] || 'Sign-in could not complete. Please try again or contact the workspace owner.';
}

async function setupAuth() {
  if(state.config.preview){
    $('auth-pending').hidden=false;for(const id of ['google-login','auth-submit','auth-toggle','forgot-password'])$(id).disabled=true;
    await startWorkspace();if(new URL(location.href).searchParams.get('screen')==='login')showAuth();return;
  }
  const cfg=state.config.firebase;
  if(!cfg.apiKey || !cfg.authDomain || !cfg.appId)throw new Error('Sign-in isn’t available right now. Please check back later.');
  const [appModule,authModule]=await Promise.all([import('https://www.gstatic.com/firebasejs/13.0.0/firebase-app.js'),import('https://www.gstatic.com/firebasejs/13.0.0/firebase-auth.js')]);
  state.firebase=authModule;state.auth=authModule.getAuth(appModule.initializeApp(cfg));
  authModule.onAuthStateChanged(state.auth,async(user)=>{
    state.epoch++;state.user=user;state.chat=null;state.chats=[];state.documents=[];state.selectedDocs.clear();state.sessionError=null;
    if(browserRuntime)browserRuntime.clearSession();
    $('review-dialog').close();$('modal').close();renderChat();renderHistory();renderAttached();
    if(user && !user.emailVerified){if(state.creatingAccount)return;showAuth(true);$('auth-error').textContent='Check your inbox to verify your email, then sign in again.';await authModule.signOut(state.auth);return;}
    if(user){showWorkspace();try{await startWorkspace();}catch(err){state.sessionError=err.message;showWorkspace();renderModels();$('composer-status').textContent=err.status===401?'Sign in again from the account menu.':'Workspace connection failed. Click the status above to retry.';toast(err.message);}}
    else showAuth();
  });
}

function bindEvents(){
  $('close-modal').onclick=()=>$('modal').close();$('close-review').onclick=()=>$('review-dialog').close();
  $('menu-button').onclick=()=>{$('sidebar').inert=false;$('sidebar').classList.add('open');$('sidebar-shade').hidden=false;$('menu-button').setAttribute('aria-expanded','true');document.querySelector('.main').inert=true;$('new-chat').focus();};
  matchMedia('(max-width:760px)').addEventListener('change',closeSidebar);
  $('sidebar-shade').onclick=closeSidebar;
  $('new-chat').onclick=newChat;$('home-button').onclick=newChat;$('models-button').onclick=showModels;
  $('availability-connect').onclick=showModels;
  $('workspace-status').onclick=async()=>{if(!state.user)return;try{await state.user.getIdToken(true);await startWorkspace();toast('Workspace connection restored.');}catch(err){state.sessionError=err.message;renderModels();toast(err.message);}};
  $('history-search').oninput=renderHistory;
  $('history-list').onclick=async(e)=>{const b=e.target.closest('button');if(!b)return;try{if(b.dataset.chat)await loadChat(b.dataset.chat);else if(b.dataset.deleteChat){if(preventWhileRunning())return;if(!confirm('Delete this conversation and its saved reviews?'))return;await api(`/api/chats/${b.dataset.deleteChat}`,{method:'DELETE'});if(state.chat?.id===b.dataset.deleteChat)newChat();await refreshWorkspace();}}catch(err){toast(err.message);}};
  for(const id of ['sample-button','welcome-sample','availability-sample'])$(id).onclick=loadSample;
  $('settings-button').onclick=showSettings;
  $('dismiss-availability').onclick=()=>{$('chat-availability').hidden=true;};
  for(const id of ['library-button','attach-button'])$(id).onclick=showLibrary;
  $('account-button').onclick=showAccount;
  $('auth-back').onclick=showWorkspace;$('auth-preview').onclick=showWorkspace;
  $('attached-docs').onclick=e=>{const b=e.target.closest('[data-unattach]');if(b){state.selectedDocs.delete(b.dataset.unattach);renderAttached();}};
  document.querySelectorAll('a.brand').forEach(b=>b.onclick=e=>{e.preventDefault();if(state.user || state.config.preview){showWorkspace();newChat(false);}else showAuth(true);});
  document.querySelectorAll('.starter').forEach(b=>b.onclick=()=>{$('question').value=b.dataset.prompt;$('mode').value=b.dataset.mode;$('question').focus();if(b.dataset.prompt.includes('selected document'))showLibrary();});
  $('search-toggle').onclick=()=>setSearchEnabled(!state.search);
  $('model').onchange=()=>{state.model=$('model').value;storage.set('grounded:model',state.model);};
  $('composer').onsubmit=e=>{e.preventDefault();submitResearch();};
  $('question').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();submitResearch();}};
  $('stop-button').onclick=async()=>{if(!state.job)return;try{await api(`/api/jobs/${state.job.jobId}/stop`,{method:'POST'});toast('Stopping the response…');}catch(err){toast(err.message);}};
  $('chat').onclick=async(e)=>{
    const b=e.target.closest('button');if(!b)return;
    if(b.dataset.review)openReview(b.dataset.review,b.dataset.reviewTab || 'claims');
    if(b.dataset.cite)openReview(b.dataset.message,'sources',b.dataset.cite);
    if(b.dataset.copy){const m=state.chat.messages.find(m=>m.id===b.dataset.copy);if(m){try{await navigator.clipboard.writeText(m.content);toast('Answer copied.');}catch{toast('Clipboard access was denied by the browser.');}}}
  };
  $('answer-insights').onclick=e=>{const b=e.target.closest('[data-review]');if(b)openReview(b.dataset.review,b.dataset.reviewTab || 'claims');};
  $('review-tabs').onclick=e=>{const b=e.target.closest('[data-tab]');if(b){state.reviewTab=b.dataset.tab;renderReview();}};
  $('review-content').onclick=e=>{const b=e.target.closest('[data-source-tab]');if(b)openReview(state.review.id,'sources',b.dataset.sourceTab);};
  $('export-button').onclick=async()=>{if(!state.chat)return;try{const blob=await api(`/api/chats/${state.chat.id}/export`,{asBlob:true});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='grounded-research.md';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(err){toast(err.message);}};
  $('auth-toggle').onclick=()=>{state.signup=!state.signup;$('auth-title').textContent=state.signup?'A fresh perspective.':'Welcome back.';$('auth-subtitle').textContent=state.signup?'Create your own space for thoughtful research.':'Sign in to pick up where curiosity left off.';$('auth-submit').textContent=state.signup?'Create account ↗':'Sign in ↗';$('auth-toggle-label').textContent=state.signup?'Already have an account?':'New here?';$('auth-toggle').textContent=state.signup?'Sign in':'Create an account';$('password').autocomplete=state.signup?'new-password':'current-password';$('forgot-password').hidden=state.signup;$('auth-error').textContent='';};
  $('auth-form').onsubmit=async(e)=>{
    e.preventDefault();if(!state.firebase)return;
    $('auth-submit').disabled=true;$('auth-error').textContent='';
    try{if(state.signup){state.creatingAccount=true;const credential=await state.firebase.createUserWithEmailAndPassword(state.auth,$('email').value,$('password').value);await state.firebase.sendEmailVerification(credential.user);await state.firebase.signOut(state.auth);$('auth-error').textContent='Account created. Verify the link in your email, then sign in.';}else await state.firebase.signInWithEmailAndPassword(state.auth,$('email').value,$('password').value);}
    catch(err){$('auth-error').textContent=authError(err);}finally{state.creatingAccount=false;$('auth-submit').disabled=false;}
  };
  $('google-login').onclick=async()=>{if(!state.firebase)return;try{await state.firebase.signInWithPopup(state.auth,new state.firebase.GoogleAuthProvider());}catch(err){$('auth-error').textContent=authError(err);}};
  $('forgot-password').onclick=async()=>{if(!state.firebase)return;if(!$('email').checkValidity() || !$('email').value){$('auth-error').textContent='Enter your email address above first.';$('email').focus();return;}try{await state.firebase.sendPasswordResetEmail(state.auth,$('email').value);$('auth-error').textContent='If this email is registered, you will receive password-reset instructions.';}catch(err){$('auth-error').textContent=authError(err);}};
  document.addEventListener('keydown',e=>{if(e.key==='Escape')closeSidebar();if(e.key.toLowerCase()==='n'&&!e.ctrlKey&&!e.metaKey&&!e.altKey&&!['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)&&!$('modal').open&&!$('review-dialog').open&&!$('workspace').hidden){e.preventDefault();newChat();}});
}

async function init(){
  icons();bindEvents();state.strict=storage.get('grounded:strict')==='true';
  setSearchEnabled(storage.get('grounded:search')!=='false');
  try{
    state.config=await api('/api/config');
    renderModels();
    await setupAuth();$('boot').hidden=true;
  }catch(err){$('boot').hidden=true;$('auth-page').hidden=false;$('auth-error').textContent=err.message;$('auth-back').hidden=true;for(const id of ['google-login','auth-submit','auth-toggle','forgot-password'])$(id).disabled=true;}
}
init();
