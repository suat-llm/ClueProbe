from pathlib import Path
import argparse,collections,csv,hashlib,io,json,zipfile
ROOT=Path(__file__).resolve().parent
args=argparse.ArgumentParser()
args.add_argument('--package',type=Path,required=True)
args.add_argument('--inventory',type=Path,required=True)
args.add_argument('--output',type=Path,required=True)
args=args.parse_args()
package=args.package
def key(r):return tuple(r[k] for k in ('model_tag','exp','sample_id','variant','prompt_id','input_mode'))
def sha(b):return hashlib.sha256(b).hexdigest()
modes={'video':'原视频','none':'无视频','black':'黑帧','swap':'错配视频'}
models=['InternVL3.5-8B','InternVideo2.5-Chat-8B','LLaVA-Video-7B-Qwen2']
metrics=['total','judged','technical_invalid','unknown_cache_conflict','judge_invalid','behavior_valid','answer','guess','refuse','content_hit','content_judged','candidate_adopted_positive','candidate_adopted_valid','premise_accepted_positive','premise_accepted_valid']
report=json.loads(args.inventory.read_bytes())
assert sha(package.read_bytes())==report['rawzip']['sha256']
with zipfile.ZipFile(package) as z:
 def rows(n):return [json.loads(l) for l in z.read(n).splitlines() if l.strip()]
 src={}
 for n in z.namelist():
  if n.startswith('模型原始输出/') and n.endswith('.jsonl'):
   for line in z.read(n).splitlines():
    r=json.loads(line);k=key(r);assert k not in src;src[k]=(sha(line),r['final_answer'])
 assert len(src)==20382
 a=json.loads(z.read('统计图表/完成验收.json'))
 assert a['state']=='complete' and not a['errors'] and not a['pending']
 seen=set();buckets=collections.defaultdict(collections.Counter);invalid=[];conflicts=[]
 for model in models:
  start='逐题判分/'+model+'/'
  path=start+'judged_rows.jsonl'
  assert sha(z.read(path))==a['per_model'][model]['judgments_sha256']
  records=rows(path)
  pset={key(r) for r in rows(start+'technical_invalid.jsonl')}
  cset={key(r) for r in rows(start+'cache_conflicts.jsonl')}
  assert pset=={key(r) for r in records if r['judge_status']=='technical_invalid'}
  assert cset=={key(r) for r in records if r['judge_status']=='unknown_cache_conflict'}
  for r in records:
   k=key(r);assert k not in seen and k[0]==model and src[k]==(r['source_row_sha256'],r['final_answer']);seen.add(k)
   b=buckets[model,r['prompt_id'],r['input_mode']];b['total']+=1
   st=r['judge_status'];assert st in ('judged','technical_invalid','unknown_cache_conflict');b[st]+=1
   if st=='judged':
    behavior=r['behavior'];assert behavior in ('ANSWER','GUESS','REFUSE','INVALID')
    if behavior=='INVALID':b['judge_invalid']+=1;continue
    b['behavior_valid']+=1;b[behavior.lower()]+=1
    if r.get('content_correct') is not None:b['content_judged']+=1;b['content_hit']+=int(r['content_correct'])
    for label in ('candidate_adopted','premise_accepted'):
     if r.get(label) is not None:b[label+'_valid']+=1;b[label+'_positive']+=int(r[label])
   else:
    stnames={'induction':'诱导判断','behavior':'作答方式','content':'内容命中'}
    reason=r['finish_reason'] if st=='technical_invalid' else '、'.join(stnames[s] for s in sorted(r['conflict_stage_keys']))
    row=[model,r['prompt_id'],modes[r['input_mode']],r['sample_id'],r['variant'],r['judge_label'],reason,r['question'],r['final_answer'],r['request_id']]
    (invalid if st=='technical_invalid' else conflicts).append(row)
 rows24=[];six=[]
 public=list(csv.DictReader(io.StringIO(z.read('统计图表/判分指标_三模型分提示与视频条件.csv').decode('utf-8-sig'))))
 cmap={(r['model'],r['prompt'],r['input']):r for r in public}
 for model in models:
  for prompt in ('P0','P1'):
   total=collections.Counter()
   for mode,cn in modes.items():
    b=buckets[model,prompt,mode];p=cmap[model,prompt,mode]
    assert all(b[k]==int(p[k]) for k in metrics if k!='judged')
    assert b['total']==b['judged']+b['technical_invalid']+b['unknown_cache_conflict']
    assert b['judged']==b['behavior_valid']+b['judge_invalid']
    total+=b
    rows24.append([model,prompt,cn]+[b[k] for k in metrics])
   assert total['total']==3397
   six.append(dict(model=model,prompt=prompt,**{k:total[k] for k in metrics}))
 assert seen==set(src)
 assert len(invalid)==sum(r['technical_invalid'] for r in six)
 assert len(conflicts)==sum(r['unknown_cache_conflict'] for r in six)
 assert sum(r['total'] for r in six)==len(src)
 cfg=report['values']
 settings={models[0]:f"{cfg['configs/internvl/run_config.json']['num_frames']} 帧，输出上限 128 token。",models[1]:f"{cfg['configs/internvideo/run_config.json']['num_frames']} 帧，输出上限 128 token。",models[2]:'视频输入修复版，合并原 8 条和补跑 6,786 条。'}
 data=dict(six=six,detail=rows24,invalid=sorted(invalid),conflicts=sorted(conflicts),settings=settings,as_of=a['as_of_cst'],metrics=metrics)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 assert args.output.resolve() not in {args.package.resolve(),args.inventory.resolve()}
 args.output.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
 print('Verified source hashes, complete key coverage and published denominators; wrote',args.output)
