"""Budgeted official-only calibration. No production outputs; retained baseline."""
import concurrent.futures,datetime,hashlib,json,threading,time,urllib.request,urllib.error
from pathlib import Path
from judge import credential,check_endpoint,NoRedirect,prompt_for,sha,validate
P=Path(__file__).parent;OUT=P/'calibration_deepseek_official';OUT.mkdir(exist_ok=True)
LOCK=threading.Lock();BUDGET=1.0
# Conservative non-cached peak CNY rates verified from official pricing 2026-10-30.
INPUT_RATE=2/1_000_000;OUTPUT_RATE=8/1_000_000
def read(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()] if p.exists() else []
def main():
 cfg=json.loads((P/'provider_deepseek_official.json').read_text(encoding='utf8'))
 check_endpoint(cfg['endpoint']);assert cfg['endpoint']=='https://api.deepseek.com/chat/completions'
 assert cfg['thinking']=={'type':'disabled'} and cfg['temperature']==0
 key=credential('DEEPSEEK_API_KEY');assert key,'Missing official key'
 t=json.loads((P/'v11_prompts.json').read_text(encoding='utf8'))
 subset=set(json.loads((P/'content_calibration_ids.json').read_text(encoding='utf8')))
 rows=read(P/'original06_inputs.jsonl');assert len(rows)==200 and len(subset)==95
 signature=sha({'config':cfg,'templates':t,'phase':'calibration','runner_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'core_sha':hashlib.sha256((P/'judge.py').read_bytes()).hexdigest()})
 jobs=[];done={};counts={};spent=0.0
 ledger=OUT/'attempt_cost_ledger.jsonl'
 for attempt in read(ledger):
  assert attempt['run_signature']==signature,'Changed calibration runner/config'
  spent+=attempt['conservative_cny']
 for stage in ['behavior','content']:
  previous=read(OUT/(stage+'.jsonl'))
  assert len({r['id'] for r in previous})==len(previous)
  assert all(r['run_signature']==signature for r in previous)
  done[stage]={r['id'] for r in previous};counts[stage]=len(previous)
  for r in rows:
   if stage=='content' and r['id'] not in subset:continue
   if r['id'] not in done[stage]:jobs.append((stage,r))
 state={'spent_conservative_cny':spent,'reserved_cny':0.0,'api_attempts_this_run':0,'budget_stopped':False}
 def status():
  data={**state,'completed':dict(counts),'budget_cny':BUDGET,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'formal_calibration_complete':counts=={'behavior':200,'content':95}}
  (OUT/'status.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
 def run(job):
  stage,r=job;prompt=prompt_for(stage,r,t);attempts=[];result=None
  # UTF8 bytes plus overhead is a deliberately loose token upper allowance.
  max_input=len(prompt.encode('utf8'))+512
  max_cost=max_input*INPUT_RATE+cfg['max_tokens']*OUTPUT_RATE
  for attempt in range(3):
   with LOCK:
    if state['spent_conservative_cny']+state['reserved_cny']+max_cost>BUDGET:
     state['budget_stopped']=True;status();return
    state['reserved_cny']+=max_cost;state['api_attempts_this_run']+=1
   body={'model':cfg['model'],'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':cfg['max_tokens'],'thinking':cfg['thinking'],'response_format':{'type':'json_object'}}
   cost=max_cost;detail={};response_id=None
   try:
    req=urllib.request.Request(cfg['endpoint'],data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.build_opener(NoRedirect()).open(req,timeout=60) as response:data=json.load(response)
    choice=data['choices'][0];answer=choice['message']['content'];usage=data.get('usage',{})
    response_id=data.get('id')
    if type(usage.get('prompt_tokens')) is int and type(usage.get('completion_tokens')) is int:
     cost=usage['prompt_tokens']*INPUT_RATE+usage['completion_tokens']*OUTPUT_RATE
    detail={'raw_output':answer,'finish_reason':choice.get('finish_reason'),'returned_model':data.get('model'),'usage':usage,'response_id':response_id}
    if choice.get('finish_reason')=='length':raise ValueError('Truncated response')
    result=json.loads(answer);validate(stage,result)
   except urllib.error.HTTPError as exc:detail={'error':'HTTPError','http_code':exc.code};result=None
   except Exception as exc:detail={**detail,'error':type(exc).__name__};result=None
   finally:
    with LOCK:
     state['reserved_cny']-=max_cost;state['spent_conservative_cny']+=cost
     record={'run_signature':signature,'id':r['id'],'stage':stage,'attempt':attempt+1,'conservative_cny':cost,'response_id':response_id,'usage':detail.get('usage')}
     with ledger.open('a',encoding='utf8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
     status()
   attempts.append(detail)
   if result is not None:break
   if attempt<2:time.sleep(2*(attempt+1))
  record={'id':r['id'],'stage':stage,'status':'ok' if result is not None else 'judge_error','result':result,'attempts':attempts,
          'input_sha':sha({'id':r['id'],'stage':stage,'prompt':prompt}),'run_signature':signature,'requested_model':cfg['model']}
  with LOCK:
   with (OUT/(stage+'.jsonl')).open('a',encoding='utf8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
   counts[stage]+=1;status()
 with LOCK:status()
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(run,jobs))
 print(json.dumps(json.loads((OUT/'status.json').read_text(encoding='utf8')),ensure_ascii=False))
if __name__=='__main__':main()
