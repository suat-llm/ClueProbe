"""v11 judge: dry-run default; explicit API config and --execute for real calls."""
import argparse,json,re,hashlib,os,time,urllib.request,urllib.parse
from pathlib import Path
P=Path(__file__).parent
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None
def credential(name):
 key=os.environ.get(name)
 if not key and os.name=='nt':
  import winreg
  try:
   with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as registry:key=winreg.QueryValueEx(registry,name)[0]
  except FileNotFoundError:pass
 return key
def check_endpoint(endpoint):
 policy_path=P/'active_provider.json'
 if policy_path.exists():
  policy=json.loads(policy_path.read_text(encoding='utf8'))
  parsed=urllib.parse.urlsplit(endpoint)
  if parsed.scheme!='https' or parsed.hostname not in policy['allowed_api_hosts'] or parsed.username or parsed.password or parsed.port not in [None,443]:
   raise ValueError('Endpoint blocked by current official-provider policy')
def read(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x.strip()]
def sha(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def prompt_for(stage,r,templates):
 values={'question':r['question'],'raw_answer':re.sub(r'^\s*(?:Answer\s*:|答案\s*[：:]|回答\s*[：:])\s*','',r['raw_answer'],flags=re.I),
 'A_or_G':r.get('label'),'reference':r.get('reference_answer',''),'guess_reasoning':r.get('guess_reasoning',''),
 'induction_type':r.get('induction_type'),'embedded_candidate_or_null':r.get('embedded_candidate'), 'presupposed_event_or_null':r.get('presupposed_event')}
 # Substitute only template placeholders in one pass; do not interpret answer text as a template.
 return re.sub(r'\{([A-Za-z_]+)\}',lambda m:str(values[m[1]]) if values[m[1]] is not None else 'null',templates[stage])
def validate(stage,result):
 if not isinstance(result,dict) or not isinstance(result.get('reason'),str):raise ValueError('Missing JSON reason')
 if stage=='behavior':
  if result.get('behavior') not in ['ANSWER','GUESS','REFUSE','INVALID']:raise ValueError('Invalid behavior')
  if 'candidate' not in result or not (result['candidate'] is None or isinstance(result['candidate'],str)):raise ValueError('Invalid candidate')
 else:
  for k in (['content_correct'] if stage=='content' else ['candidate_adopted','premise_accepted']):
   if type(result.get(k)) is not int or result[k] not in [0,1]:raise ValueError('Invalid '+k)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['behavior','content','induction'],required=True)
 ap.add_argument('--input',default=str(P/'original06_inputs.jsonl'));ap.add_argument('--output',required=True)
 ap.add_argument('--phase',choices=['calibration','production'],default='calibration');ap.add_argument('--behavior-file')
 ap.add_argument('--config');ap.add_argument('--execute',action='store_true');ap.add_argument('--max-requests',type=int,default=400)
 ap.add_argument('--selection-ids',help='Fixed calibration subset IDs')
 a=ap.parse_args();templates=json.loads((P/'v11_prompts.json').read_text(encoding='utf8'));rows=read(a.input)
 assert len({r['id'] for r in rows})==len(rows)
 if a.selection_ids:
  if a.phase!='calibration':raise ValueError('Fixed subset only supported for calibration')
  selected=json.loads(Path(a.selection_ids).read_text(encoding='utf8'))
  if len(selected)!=len(set(selected)) or not set(selected)<={r['id'] for r in rows}:raise ValueError('Invalid selection IDs')
  rows=[r for r in rows if r['id'] in set(selected)]
 if a.stage=='content':
  rows=[r for r in rows if r['label'] in ['A','G']]
  if a.phase=='production':
   if not a.behavior_file:raise ValueError('Production content needs frozen judge behavior output')
   beh={r['id']:r for r in read(a.behavior_file)}
   if any(r['id'] not in beh or beh[r['id']].get('status')!='ok' for r in rows):raise ValueError('Resolve missing/failed behavior before content selection')
   rows=[r for r in rows if beh[r['id']]['result']['behavior'] in ['ANSWER','GUESS']]
 if a.stage=='induction':
  rows=[r for r in rows if r.get('variant') in ['v2','v3','v5','u1']]
 output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
 prompts=[{'id':r['id'],'stage':a.stage,'prompt':prompt_for(a.stage,r,templates)} for r in rows]
 if not a.execute:
  output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in prompts),encoding='utf8');print('DRY_RUN',len(prompts),'no API called');return
 if not a.config:raise ValueError('Explicit provider/model configuration required')
 cfg=json.loads(Path(a.config).read_text(encoding='utf8'))
 assert cfg['temperature']==0 and cfg['model'] and cfg['endpoint'].startswith('https://')
 check_endpoint(cfg['endpoint'])
 key=credential(cfg['api_key_env']);assert key,'API credential environment variable missing'
 signature=sha({'config':cfg,'templates':templates,'phase':a.phase,'code_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 prior=read(output) if output.exists() else [];assert all(r['run_signature']==signature for r in prior)
 done={r['id']:r for r in prior};assert len(done)==len(prior)
 assert set(done)<={r['id'] for r in rows}
 calls=0
 with output.open('a',encoding='utf8') as f:
  for r in prompts:
   input_sha=sha(r)
   if r['id'] in done:
    assert done[r['id']]['input_sha']==input_sha
    continue
   attempts=[];result=None
   for attempt in range(3):
    if calls>=a.max_requests:raise RuntimeError('API request cap reached; completed records preserved')
    calls+=1
    body={'model':cfg['model'],'messages':[{'role':'user','content':r['prompt']}],'temperature':0,'max_tokens':cfg.get('max_tokens',256),'response_format':{'type':'json_object'}}
    if 'thinking' in cfg:body['thinking']=cfg['thinking']
    try:
     req=urllib.request.Request(cfg['endpoint'],data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
     with urllib.request.build_opener(NoRedirect()).open(req,timeout=120) as response:raw=json.load(response)
     choice=raw['choices'][0];text=choice['message']['content'];attempts.append({'raw_output':text,'finish_reason':choice.get('finish_reason'),'returned_model':raw.get('model'),'fingerprint':raw.get('system_fingerprint'),'usage':raw.get('usage')})
     if choice.get('finish_reason')=='length':raise ValueError('Truncated judge response')
     result=json.loads(text);validate(a.stage,result);break
    except Exception as exc:
     attempts.append({'error':type(exc).__name__,'message':str(exc)[:500]});result=None
     if attempt<2:time.sleep(2*(attempt+1))
   record={'id':r['id'],'stage':a.stage,'status':'ok' if result is not None else 'judge_error','result':result,'attempts':attempts,'input_sha':input_sha,'run_signature':signature,'requested_model':cfg['model']}
   f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
 print('FINISHED',len(rows),'API attempts this invocation',calls)
if __name__=='__main__':main()
