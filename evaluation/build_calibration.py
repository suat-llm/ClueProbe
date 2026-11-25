"""New 200-answer human calibration pack, not the missing old annotation06 export."""
import csv,json,random,html,collections,hashlib
from pathlib import Path
P=Path(__file__).parent;B=P.parent;O=P/'calibration200';O.mkdir(exist_ok=True)
with (B/'video_ooc_six_model_integrated_20261029/六模型_逐题合并_20646条.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
def read(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
ll={(str(r['task_id']),r['prompt_id']):r for r in read(B/'llava7b_chinese_audit_20261029/raw_predictions_LLaVA-Video-7B.jsonl')}
vl={(str(r['task_id']),r['prompt_id']):r for r in read(B/'video_ooc_project23_videollama3_rerun_20261026/results/raw_predictions_VideoLLaMA3-7B.jsonl')}
for r in rows:
 k=(r['task_id'],r['prompt_id'])
 if r['model']=='LLaVA-Video-7B':
  x=ll[k];assert r['question']==x['question'];r.update(model_answer=x['new_answer'],protocol_version=x['protocol_version'])
 if r['model']=='VideoLLaMA3-7B':
  x=vl[k];assert r['question']==x['question'];r.update(model_answer=x['raw_answer'],protocol_version=x['protocol_version'])
rng=random.Random(20261030);groups=collections.defaultdict(list)
for r in rows:groups[(r['model'],r['prompt_id'],r['label'])].append(r)
for g in groups.values():rng.shuffle(g)
keys=sorted(groups);rng.shuffle(keys);selected=[]
while len(selected)<200:
 for k in keys:
  if len(selected)>=200:break
  selected.append(groups[k].pop())
rng.shuffle(selected)
for i,r in enumerate(selected):r.update(calibration_id=f'CAL_{i+1:03d}',human_behavior=None,human_content_correct=None,human_reason=None,human_review_status='pending')
(O/'calibration200_source.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf8')
esc=lambda x:html.escape(str(x or ''))
content=['''<!doctype html><meta charset="utf-8"><title>六模型裁判校准200条</title><style>body{max-width:960px;margin:30px auto;font:16px/1.65 system-ui}article{border:1px solid #ccc;border-radius:10px;padding:18px;margin:18px 0}textarea{width:95%;min-height:60px}button,select{padding:8px}pre{white-space:pre-wrap}.bar{position:sticky;top:0;background:white;padding:12px}</style><h1>六模型人工校准：200条待标</h1><p>全新抽样，不是手册所称已完成的标注06。六模型×P0/P1×A/G/U分层；LLaVA与VideoLLaMA3使用修复版回答。请先独立判断方式，再展开内容参考。模型身份隐藏，无预填AI结论。不确定就标待复核，不猜判。点击导出保存JSON；页面自动暂存在本机浏览器，刷新可恢复（同一文件）。</p><p>ANSWER：确定给出候选；GUESS：给候选并明确不确定；REFUSE：无法确定且无候选；INVALID：空、乱码或无关。先拒后猜算GUESS。A内容按核心语义与细度；G按合理候选与可见线索，参考不是唯一答案；U或拒答的内容选不适用。线索不足时选待复核。</p><div class="bar"><button onclick="save()">导出人工标注JSON</button> <span id="progress"></span></div>''']
for i,r in enumerate(selected):
 content.append(f'''<article data-i="{i}"><b>{r['calibration_id']}</b><p>问题：{esc(r['question'])}</p><p>模型回答：{esc(r['model_answer'])}</p><label>回答方式 <select name="behavior"><option value="">未判</option><option>ANSWER</option><option>GUESS</option><option>REFUSE</option><option>INVALID</option><option value="review">待复核</option></select></label><details><summary>方式判定后展开参考信息</summary><p>原题标签：{esc(r['label'])}；参考答案：{esc(r['reference_answer'])}；参考猜测：{esc(r['reference_guess'])}</p><p>源备注：{esc(r['annotation_note'])}</p><p>视频ID：{esc(r['video_id'])}；路径：{esc(r['source_video_path'])}</p></details><p>内容 <select name="content"><option value="">未判</option><option value="1">正确／可接受</option><option value="0">错误</option><option value="na">不适用（U／拒答／无效）</option><option value="review">待复核</option></select></p><textarea name="reason" placeholder="判断理由／候选答案／需要核视频的问题"></textarea></article>''')
content.append('''<script>const key='video_ooc_calibration200_20261030';const source=DATA;function collect(){return [...document.querySelectorAll('article')].map(el=>({...source[Number(el.dataset.i)],human_behavior:el.querySelector('[name=behavior]').value,human_content_correct:el.querySelector('[name=content]').value,human_reason:el.querySelector('[name=reason]').value}));}function changed(){let r=collect();localStorage.setItem(key,JSON.stringify(r));document.getElementById('progress').textContent='已选回答方式 '+r.filter(x=>x.human_behavior).length+'/200';}try{let prior=JSON.parse(localStorage.getItem(key)||'[]');document.querySelectorAll('article').forEach((el,i)=>{if(prior[i]&&prior[i].calibration_id===source[i].calibration_id){el.querySelector('[name=behavior]').value=prior[i].human_behavior||'';el.querySelector('[name=content]').value=prior[i].human_content_correct||'';el.querySelector('[name=reason]').value=prior[i].human_reason||'';}});}catch(e){}document.addEventListener('change',changed);document.addEventListener('input',changed);function save(){changed();let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(collect(),null,2)],{type:'application/json'}));a.download='六模型_200条人工校准_回填.json';a.click();}changed();</script>'''.replace('DATA',json.dumps(selected,ensure_ascii=False).replace('</','<\\/')))
(O/'六模型_200条人工校准.html').write_text(''.join(content),encoding='utf8')
(O/'manifest.json').write_text(json.dumps({'n':len(selected),'seed':20261030,'status':'all_pending_human','sampling':'round-robin randomized stratified model/prompt/label; 5-6 per36strata; not proportional prevalence estimate','counts':dict(collections.Counter(r['model'] for r in selected)),'source_sha256':hashlib.sha256((O/'calibration200_source.jsonl').read_bytes()).hexdigest()},ensure_ascii=False,indent=2),encoding='utf8')
print('Created 200 pending human records')
