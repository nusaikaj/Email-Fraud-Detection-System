const form=document.querySelector('#scanForm');
const file=document.querySelector('#file');
const el=id=>document.getElementById(id);
file.addEventListener('change',()=>{el('filename').textContent=file.files[0]?.name||'No file selected'});
form.addEventListener('submit',async e=>{
 e.preventDefault();el('error').hidden=true;
 const btn=el('scan');btn.disabled=true;btn.textContent='Analyzing…';
 const data=new FormData();if(file.files.length){data.append('file',file.files[0]);}else{data.append('email',el('email').value)}
 try{
  const res=await fetch('/api/analyze',{method:'POST',body:data});
  const result=await res.json();if(!res.ok)throw new Error(result.error||'Analysis failed');
  el('placeholder').hidden=true;el('results').hidden=false;
  el('label').textContent=result.label;
  el('score').textContent=`Threat model probability (uncalibrated): ${(result.threat_probability*100).toFixed(1)}%`;
  el('meterfill').style.width=`${result.threat_probability*100}%`;
  el('meterfill').style.background=result.threat_probability>=.5?'#fb866d':'#39d6b8';
  const populate=(id,values,empty)=>{const list=el(id);list.replaceChildren();(values.length?values:[empty]).forEach(item=>{const li=document.createElement('li');li.textContent=item;list.append(li)})};
  populate('signals',result.signals,'No rule-based warning signals detected');
  populate('links',result.links_found,'No HTTP(S) links detected');
  el('disclaimer').textContent=result.disclaimer;
  await refreshStats();await refreshOverview();
 }catch(error){el('error').textContent=error.message;el('error').hidden=false}
 finally{btn.disabled=false;btn.textContent='◎  Analyze email'}
});

async function refreshStats(){try{const res=await fetch('/api/stats');if(!res.ok)return;const r=await res.json();el('totalScans').textContent=r.total;el('spamScans').textContent=r.threats;el('flaggedScans').textContent=r.flagged}catch(e){console.warn('Stats unavailable',e)}}
async function refreshHistory(){try{const res=await fetch('/api/history');if(!res.ok)throw new Error('History unavailable');const r=await res.json();const tbody=el('historyRows');tbody.replaceChildren();if(!r.scans.length){const tr=document.createElement('tr');const td=document.createElement('td');td.colSpan=6;td.textContent='No scans yet.';tr.append(td);tbody.append(tr);return}for(const scan of r.scans){const tr=document.createElement('tr');for(const value of [scan.scanned_at,scan.label,(scan.score*100).toFixed(1)+'%',scan.signal_count,scan.link_count,scan.model_version]){const td=document.createElement('td');td.textContent=value;tr.append(td)}tbody.append(tr)}}catch(e){console.warn(e);el('historyRows').textContent='Unable to load history.'}}
for(const nav of document.querySelectorAll('[data-page]'))nav.addEventListener('click',()=>{const page=nav.dataset.page;document.querySelectorAll('[data-page]').forEach(n=>n.classList.toggle('active',n===nav));document.querySelectorAll('.page').forEach(p=>{p.hidden=p.id!==`page-${page}`});if(page==='history')refreshHistory();if(page==='overview')refreshOverview();if(page==='performance')refreshPerformance();document.querySelector('.top span').textContent='SECURITY WORKSPACE / '+page.toUpperCase();document.body.dataset.currentPage=page});
el('refreshHistory').addEventListener('click',refreshHistory);
refreshStats();

// V3.3 data-driven analytics: no third-party libraries or network calls.
const fmt=n=>Number(n||0).toLocaleString();
async function apiJson(url){const res=await fetch(url,{cache:'no-store'});if(!res.ok)throw Error('Could not load '+url);return res.json()}
async function refreshOverview(){
 try{
  const a=await apiJson('/api/analytics');
  for(const [id,val] of Object.entries({oScans:a.current_model_scans,oThreat:a.threat,oSafe:a.safe,oFlags:a.flagged,oLinks:a.linked,oLegacy:a.legacy_scans,mixThreat:a.threat,mixSafe:a.safe,mixLegacy:a.legacy_scans}))el(id).textContent=fmt(val);
  const pct=a.current_model_scans?(100*a.threat/a.current_model_scans):0;
  el('mixPercent').textContent=a.current_model_scans?pct.toFixed(1)+'%':'—';
  el('mixDonut').style.background=a.current_model_scans?`conic-gradient(#fb897d 0 ${pct}%,#31d4b7 ${pct}% 100%)`:'conic-gradient(#263b51 0 100%)';
  drawTrend(a.daily);
  const metric=await apiJson('/api/model-metrics');drawModelCards(metric.holdout);
 }catch(err){el('trendChart').textContent='Analytics unavailable: '+err.message;}
}
function drawTrend(days){
 const root=el('trendChart');root.replaceChildren();
 if(!days.length){const p=document.createElement('p');p.className='chart-empty';p.textContent='No V3.2 scans yet. Analyze an email to begin tracking.';root.append(p);return}
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 700 220');svg.setAttribute('role','img');svg.setAttribute('aria-label','Threat and safe scans per day');
 const mk=(tag,attrs,content)=>{let n=document.createElementNS(svg.namespaceURI,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(content!==undefined)n.textContent=content;svg.append(n);return n};
 const max=Math.max(1,...days.map(d=>d.safe+d.threat));const w=620/days.length;const base=176;
 for(let t=0;t<=4;t++){const y=base-t*36;mk('line',{x1:43,y1:y,x2:682,y2:y,stroke:'#263c51','stroke-width':1});mk('text',{x:5,y:y+4,fill:'#7798b6','font-size':10},String(Math.round(max*t/4)))}
 days.forEach((d,i)=>{const x=48+i*w;const bh=d.safe/max*140;const th=d.threat/max*140;mk('rect',{x,y:base-bh,width:Math.max(4,w-14),height:bh,rx:2,fill:'#31d4b7'});mk('rect',{x,y:base-bh-th,width:Math.max(4,w-14),height:th,rx:2,fill:'#fb897d'});if(i%Math.max(1,Math.ceil(days.length/7))===0)mk('text',{x:x,y:205,fill:'#8faec7','font-size':10},d.date.slice(5))});root.append(svg)
}
function drawModelCards(data){const root=el('modelCards');root.replaceChildren();if(!data){root.textContent='Final holdout report not found. Copy reports/grouped/final_test_results.json into the project.';return}for(const [label,key] of [['Accuracy','accuracy'],['Threat precision','precision'],['Threat recall','recall'],['Threat F1','f1_score']]){const div=document.createElement('div');div.className='metric-tile';const small=document.createElement('small');small.textContent=label;const strong=document.createElement('strong');strong.textContent=(100*data[key]).toFixed(2)+'%';div.append(small,strong);root.append(div)}}
// V3.4: factual, client-rendered performance visuals from existing local reports.
function node(tag,cls,text){const x=document.createElement(tag);if(cls)x.className=cls;if(text!==undefined)x.textContent=String(text);return x}
function metricCard(title,value,desc){const b=node('div','v34-kpi');b.append(node('span','v34-kpi-label',title),node('strong','v34-kpi-value',value),node('small','v34-kpi-desc',desc));return b}
function modelPercent(v){return Number.isFinite(Number(v))?(Number(v)*100).toFixed(2)+'%':'—'}
function renderMatrix(cm){const card=node('section','panel v34-matrix-card');card.append(node('h3','', 'Final test confusion matrix'),node('p','helper','Rows = actual class · columns = predicted class'));if(!Array.isArray(cm)||cm.length!==2||cm.some(r=>!Array.isArray(r)||r.length!==2)){card.append(node('p','helper','Confusion matrix unavailable.'));return card}
 const grid=node('div','v34-confusion');for(const name of ['','Predicted safe','Predicted threat'])grid.append(node('div','v34-matrix-head',name));
 for(let r=0;r<2;r++){grid.append(node('div','v34-matrix-head',r===0?'Actual safe':'Actual threat'));for(let c=0;c<2;c++){const x=node('div',r===c?'v34-cm-cell correct':'v34-cm-cell incorrect');x.append(node('strong','',Number(cm[r][c]).toLocaleString()),node('small','',r===0&&c===0?'True negative':r===0?'False positive':c===0?'False negative':'True positive'));grid.append(x)}}card.append(grid);return card}
function comparisonCard(rows){const card=node('section','panel v34-compare');card.append(node('h3','','Validation · model comparison'),node('p','helper','F1 and recall on grouped validation. Random Forest used fewer training samples.'));
 if(!Array.isArray(rows)||!rows.length){card.append(node('p','helper','Validation report unavailable.'));return card}
 for(const r of rows){const wrap=node('div','v34-model-row');const head=node('div','v34-model-row-head');head.append(node('strong','',r.model.replaceAll('_',' ')),node('span','',`F1 ${modelPercent(r.f1_score)} · Recall ${modelPercent(r.recall)}`));wrap.append(head);const track=node('div','v34-bar-track');const fill=node('div','v34-bar-fill');fill.style.width=Math.max(0,Math.min(100,Number(r.f1_score)*100))+'%';track.append(fill);wrap.append(track);card.append(wrap)}return card}
async function refreshPerformance(){const root=el('performanceContent');try{const a=await apiJson('/api/model-metrics');root.replaceChildren();if(!a.holdout){root.append(node('p','helper','Final grouped holdout report not found under reports/grouped/'));return}const h=a.holdout;
 const stats=node('div','v34-metrics');for(const [title,key,desc] of [['Accuracy','accuracy','All classes'],['Precision','precision','Threat class'],['Recall','recall','Threat detection'],['F1 score','f1_score','Threat class']])stats.append(metricCard(title,modelPercent(h[key]),desc));root.append(stats);
 const two=node('div','v34-two');two.append(renderMatrix(h.confusion_matrix),comparisonCard(a.validation));root.append(two);
 const foot=node('section','panel v34-notes');foot.append(node('h3','','Evaluation context'),node('p','helper',`Final grouped holdout · ${Number(h.test_samples||0).toLocaleString()} emails · ${h.false_negatives} missed threats · ${h.false_positives} false alarms. Validation is used for model selection; final holdout is reported separately.`),node('p','disclaimer',a.note||'Mixed threat-email classification; not phishing-only or independently externally validated. Scores are not calibrated real-world risk.'));root.append(foot)
 }catch(e){root.replaceChildren(node('p','helper','Performance data unavailable: '+e.message))}}
el('refreshOverview').addEventListener('click',refreshOverview);el('startAnalyze').addEventListener('click',()=>document.querySelector('[data-page="analyzer"]').click());
refreshOverview();

document.body.dataset.currentPage='overview';
