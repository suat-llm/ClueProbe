import argparse,json,collections,hashlib
from pathlib import Path
P=Path(__file__).parent
def kappa(a,b):
 if not a:return None
 ca,cb=collections.Counter(a),collections.Counter(b);n=len(a);po=sum(x==y for x,y in zip(a,b))/n
 pe=sum(ca[k]*cb[k] for k in set(ca)|set(cb))/(n*n)
 return (po-pe)/(1-pe) if pe<1 else None
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--human',required=True);ap.add_argument('--behavior',required=True);ap.add_argument('--content',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 h=json.loads(Path(a.human).read_text(encoding='utf8'));manifest=json.loads((P/'manifest.json').read_text(encoding='utf8'))
 assert h['source_sha256']==manifest['source_sha256'] and h.get('annotator','').strip()
 assert h.get('provenance') in ['human_entry_blank_original06_ui','existing_exported_baseline_teacher_described_human'],'Unknown baseline provenance'
 rows=[json.loads(x) for x in (P/'original06_inputs.jsonl').read_text(encoding='utf8').splitlines()];expected={r['id']:r for r in rows}
 human={r['id']:r for r in h['records']};assert len(human)==len(h['records'])==200 and set(human)==set(expected)
 def load(path):
  rs=[json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines()];d={r['id']:r for r in rs};assert len(d)==len(rs);return d
 b,c=load(a.behavior),load(a.content);hb=[];pb=[];hc=[];pc=[];missing=[];pairs=[]
 from judge import prompt_for,sha,validate
 templates=json.loads((P/'v11_prompts.json').read_text(encoding='utf8'))
 signatures={r['run_signature'] for r in list(b.values())+list(c.values())};assert len(signatures)==1,'Mixed judge configurations'
 for stage,data in [('behavior',b),('content',c)]:
  for id,r in data.items():
   assert id in expected and r['stage']==stage
   assert r['input_sha']==sha({'id':id,'stage':stage,'prompt':prompt_for(stage,expected[id],templates)}),'Input mismatch'
   if r['status']=='ok':validate(stage,r['result'])
 for id,r in human.items():
  if r['behavior'] not in ['ANSWER','GUESS','REFUSE','INVALID']:missing.append([id,'human_behavior']);continue
  if id not in b or b[id]['status']!='ok':missing.append([id,'judge_behavior']);continue
  hb.append(r['behavior']);pb.append(b[id]['result']['behavior'])
  applicable=expected[id]['label'] in ['A','G'] and r['behavior'] in ['ANSWER','GUESS']
  if applicable:
   if r['content'] not in ['0','1']:missing.append([id,'human_content']);continue
   if id not in c or c[id]['status']!='ok':missing.append([id,'judge_content']);continue
   hc.append(int(r['content']));pc.append(c[id]['result']['content_correct'])
  pairs.append({'id':id,'human':r,'judge_behavior':b[id]['result'],'judge_content':c.get(id,{}).get('result') if applicable else None})
 kb,kc=kappa(hb,pb),kappa(hc,pc)
 report={'behavior_n':len(hb),'content_n':len(hc),'behavior_kappa':kb,'content_kappa':kc,'behavior_agreement':sum(x==y for x,y in zip(hb,pb))/len(hb) if hb else None,'missing_or_unresolved':missing,'thresholds_passed':not missing and len(hb)==200 and kb is not None and kc is not None and kb>=.8 and kc>=.7,'baseline_provenance':h['provenance'],'note':'Existing exported judgments are the teacher-described baseline. Identity not independently verified. One candidate calibration only; candidate comparison and production scoring are separate.'}
 import csv,re
 def language(answer):
  answer=re.sub(r'^\s*(?:Answer\s*:|答案\s*[：:]|回答\s*[：:])\s*','',answer,flags=re.I)
  zh=bool(re.search(r'[\u4e00-\u9fff]',answer));en=bool(re.search('[A-Za-z]',answer))
  return 'mixed' if zh and en else 'zh' if zh else 'en' if en else 'other'
 groups=collections.defaultdict(lambda:([],[]))
 flat=[]
 for pair in pairs:
  id=pair['id'];lang=language(expected[id]['raw_answer'])
  if pair['judge_content'] is not None:
   groups[lang][0].append(int(pair['human']['content']));groups[lang][1].append(pair['judge_content']['content_correct'])
  flat.append({'id':id,'question':expected[id]['question'],'model_answer':expected[id]['raw_answer'],'reference_answer':expected[id]['reference_answer'],'answer_lang':lang,'human_behavior':pair['human']['behavior'],'human_content':pair['human']['content'],'judge_behavior':pair['judge_behavior']['behavior'],'judge_behavior_reason':pair['judge_behavior']['reason'],'judge_content':pair['judge_content']['content_correct'] if pair['judge_content'] else None,'judge_content_reason':pair['judge_content']['reason'] if pair['judge_content'] else None})
 report['content_by_language']={lang:{'n':len(groups[lang][0]),'kappa':kappa(*groups[lang]),'agreement':sum(x==y for x,y in zip(*groups[lang]))/len(groups[lang][0]) if groups[lang][0] else None} for lang in ['zh','en','mixed','other']}
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'calibration_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');(out/'comparison.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs),encoding='utf8')
 with (out/'judge_calibration.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['id','question','model_answer','reference_answer','answer_lang','human_behavior','human_content','judge_behavior','judge_behavior_reason','judge_content','judge_content_reason']);w.writeheader();w.writerows(flat)
 print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
