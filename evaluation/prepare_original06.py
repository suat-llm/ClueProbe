import json,hashlib,html
from pathlib import Path
from html.parser import HTMLParser
P=Path(__file__).parent;P.mkdir(exist_ok=True)
src=Path(r'C:\Users\来去匆匆\Downloads\project-47-at-2026-10-30-08-07-c0826af3.json')
manual=Path(r'D:\实验\OneDrive\Desktop\科研\VideoOOC_实验指导手册v11_2026-10-26.html')
tasks=json.loads(src.read_text(encoding='utf8'));assert len(tasks)==200
rows=[]
for t in tasks:
 d=t['data'];rows.append({'id':d['item'],'task_id':t['id'],'sample_id':d['sample_id'],'question':d['question'],'raw_answer':d['model_answer'],'label':d['reference_type'][0],'reference_type':d['reference_type'],'reference_answer':d.get('reference_answer',''),'guess_reasoning':d.get('guess_reasoning','')})
assert len({r['id'] for r in rows})==200
digest=hashlib.sha256(src.read_bytes()).hexdigest()
(P/'original06_inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
class Pre(HTMLParser):
 def __init__(self):super().__init__();self.on=False;self.buf=[];self.blocks=[]
 def handle_starttag(self,t,a):
  if t=='pre':self.on=True;self.buf=[]
 def handle_endtag(self,t):
  if t=='pre':self.on=False;self.blocks.append(''.join(self.buf).strip())
 def handle_data(self,d):
  if self.on:self.buf.append(d)
p=Pre();p.feed(manual.read_text(encoding='utf8'))
prompts={}
for stage,needle in [('behavior','你是视频问答评测中的'),('content','你是宽松的语义判分员'),('induction','candidate_adopted')]:
 candidates=[b for b in p.blocks if needle in b]
 assert len(candidates)==1,(stage,len(candidates));prompts[stage]=candidates[0]
(P/'v11_prompts.json').write_text(json.dumps(prompts,ensure_ascii=False,indent=2),encoding='utf8')
esc=lambda x:html.escape(str(x or ''))
parts=['''<!doctype html><html lang="zh"><meta charset="utf-8"><title>标注06原始200条人工判定</title><style>body{max-width:960px;margin:30px auto;font:16px/1.7 system-ui;background:#f7f9fc;color:#17243b}article{background:white;border:1px solid #ccd4df;padding:20px;margin:18px 0;border-radius:10px}textarea{width:95%;min-height:55px}button,input,select{padding:10px;margin:5px}.bar{position:sticky;top:0;background:#eaf1ff;padding:10px}details{margin:14px 0}pre{white-space:pre-wrap}</style><h1>标注06：原始200条人工判定</h1><p>来自你提供的Project47导出，保留原题与原回答。旧预填判断未载入本页面，全部从空白开始。先判回答方式，再展开参考判断内容。不要把参考答案的正确性当成模型回答方式。</p><p>ANSWER：确定给出候选；GUESS：给候选且明确不确定；REFUSE：表示无法确定且不给候选；INVALID：空、乱码或无关。先拒后给候选算GUESS；候选需达到问题所问细度。A内容按核心语义；G候选与参考同类且不违背画面线索即可，不要求字面一致。拿不准选待复核。</p><div class="bar">标注人代号：<input id="annotator" placeholder="如JW或student1"><button onclick="save()">导出人工判定</button><span id="progress"></span></div><p>每次结束请导出备份。页面会尝试本机暂存；两人独立标注请使用各自浏览器或电脑。原题标签和参考只在展开后显示。</p>''']
for i,r in enumerate(rows):
 parts.append(f'''<article data-i="{i}"><b>{esc(r['id'])} · {i+1}/200</b><p>问题：{esc(r['question'])}</p><pre>模型回答：{esc(r['raw_answer'])}</pre><label>回答方式 <select name="behavior"><option value="">未判定</option><option>ANSWER</option><option>GUESS</option><option>REFUSE</option><option>INVALID</option><option value="review">待复核</option></select></label><details><summary>完成方式判断后，展开参考信息</summary><p>{esc(r['reference_type'])}</p><p>参考答案：{esc(r['reference_answer'])}</p><p>画面线索：{esc(r['guess_reasoning'])}</p><p>原题编号：{esc(r['sample_id'])}</p></details><label>内容判分 <select name="content"><option value="">未判定</option><option value="1">1 · 正确/合理</option><option value="0">0 · 错误</option><option value="na">不适用（U或拒答/无效）</option><option value="review">待复核</option></select></label><textarea name="reason" placeholder="判断理由或待复核问题"></textarea></article>''')
js='''<script>const rows=ROWS,sha=SHA,key='original06_human_v1_'+sha.slice(0,12);function collect(){return [...document.querySelectorAll('article')].map(el=>({id:rows[+el.dataset.i].id,behavior:el.querySelector('[name=behavior]').value,content:el.querySelector('[name=content]').value,reason:el.querySelector('[name=reason]').value}));}function update(){const obj={source_sha256:sha,annotator:document.getElementById('annotator').value,records:collect()};try{localStorage.setItem(key,JSON.stringify(obj));}catch(e){}document.getElementById('progress').textContent='方式已选 '+obj.records.filter(r=>r.behavior).length+'/200';return obj;}try{const old=JSON.parse(localStorage.getItem(key)||'null');if(old&&old.source_sha256===sha){document.getElementById('annotator').value=old.annotator||'';document.querySelectorAll('article').forEach((el,i)=>{let r=old.records.find(r=>r.id===rows[i].id);if(r){el.querySelector('[name=behavior]').value=r.behavior;el.querySelector('[name=content]').value=r.content;el.querySelector('[name=reason]').value=r.reason;}});}}catch(e){}function save(){let obj=update();if(!obj.annotator.trim()){alert('请先填写标注人代号');return;}obj.saved_at=new Date().toISOString();obj.provenance='human_entry_blank_original06_ui';let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(obj,null,2)],{type:'application/json'}));a.download='标注06_原200条_人工回填.json';a.click();}document.addEventListener('input',update);document.addEventListener('change',update);update();</script></html>'''
parts.append(js.replace('ROWS',json.dumps(rows,ensure_ascii=False).replace('</','<\\/')).replace('SHA',json.dumps(digest)))
(P/'标注06_原200条_空白人工判定.html').write_text(''.join(parts),encoding='utf8')
manifest={'source_file':str(src),'source_sha256':digest,'n':200,'human_gold_status':'existing_exported_baseline_teacher_described_human','original_prefilled_values_not_used':False,'independently_verified_human_provenance':False,'manual_path':str(manual),'manual_sha256':hashlib.sha256(manual.read_bytes()).hexdigest(),'prompts_sha256':hashlib.sha256((P/'v11_prompts.json').read_bytes()).hexdigest()}
(P/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print('Prepared original200, blank human UI, exact v11 prompts')
