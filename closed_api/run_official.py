"""Portable LADDER/INDUCE relay runner; explicit frozen media/generation protocol."""
import argparse,concurrent.futures as cf,collections,datetime,fcntl,getpass,hashlib,json,os,subprocess,sys,tempfile,threading,time,re
from pathlib import Path
parser=argparse.ArgumentParser(description='Official Google/OpenAI API adapter. Defaults to offline validation; --execute enables paid calls.')
parser.add_argument('--run-root',type=Path,help='Prepared run root containing data/ and datasets/')
parser.add_argument('--output-root',type=Path,help='Separate writable output directory')
parser.add_argument('--provider',choices=['google','openai'],required=True)
parser.add_argument('--model',required=True,help='Select one independently resumed model')
parser.add_argument('--execute',action='store_true',help='Allow paid API calls; otherwise validate only')
parser.add_argument('--stage',choices=['smoke','pipeline'],help='Smoke only, or smoke followed by remaining full input')
parser.add_argument('--experiments',nargs='+',choices=['LADDER','INDUCE'])
parser.add_argument('--validate-only',action='store_true',help='Check paths without reading a key or calling an API')
args=parser.parse_args()
for field,var in [('run_root','VIDEOOOC_PACKAGE_ROOT'),('output_root','VIDEOOOC_OUTPUT_ROOT'),('model','VIDEOOOC_MODEL_FILTER'),('stage','VIDEOOOC_STAGE')]:
 if getattr(args,field) is not None:os.environ[var]=str(getattr(args,field))
if args.experiments:os.environ['VIDEOOOC_EXPERIMENTS']=','.join(args.experiments)
if not args.execute:args.validate_only=True
if args.validate_only:os.environ['VIDEOOOC_VALIDATE_ONLY']='1'
if 'VIDEOOOC_OUTPUT_ROOT' not in os.environ:parser.error('Provide --output-root or VIDEOOOC_OUTPUT_ROOT')
ROOT=Path(os.environ.get('VIDEOOOC_PACKAGE_ROOT',str(Path(__file__).resolve().parents[2]))).resolve()
OUT=Path(os.environ['VIDEOOOC_OUTPUT_ROOT']).expanduser().resolve()
assert OUT != ROOT and ROOT not in OUT.parents, 'Write outputs in your own directory outside the shared handoff'
SRC=ROOT
OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
BASE='https://generativelanguage.googleapis.com' if args.provider=='google' else 'https://api.openai.com'
MODELS=[args.model];ACTIVE_MODELS=MODELS
from prompts import PROMPTS
from manifest import smoke_jobs, validate_jobs
jobs=[json.loads(s) for s in (SRC/'data/inference_jobs.jsonl').read_text().splitlines() if s.strip()];EXPERIMENTS=set(os.environ.get('VIDEOOOC_EXPERIMENTS','LADDER,INDUCE').split(','));assert EXPERIMENTS <= {'LADDER','INDUCE'} and EXPERIMENTS
jobs=[j for j in jobs if j['exp'] in EXPERIMENTS];TARGET=len(jobs);assert TARGET>0, 'Empty selected experiment manifest'
PRICING_FILE=os.environ.get('VIDEOOOC_PRICING_JSON');PRICES=json.loads(Path(PRICING_FILE).read_text()) if PRICING_FILE else {}
STAGE=os.environ.get('VIDEOOOC_STAGE','smoke');assert STAGE in {'smoke','pipeline'}
ALIAS_FILE=None;ALIASES={}
if ALIAS_FILE:
 approval=json.loads(Path(ALIAS_FILE).read_text());assert approval.get('confirmed_by') and approval.get('confirmed_at') and approval.get('evidence'), 'Alias approval must include confirmation evidence'
 ALIASES=approval['accepted_returned_models'];assert set(ALIASES)<=set(MODELS)
def accepted_model(tag,returned):return returned==tag
validate_jobs(jobs, ROOT)
smoke=smoke_jobs(jobs);SMOKE_TARGET=len(smoke)
key=lambda j:tuple(j[x] for x in ['exp','sample_id','variant','prompt_id','input_mode'])
EXPECTED={key(j):j for j in jobs};assert len(EXPECTED)==TARGET
for j in jobs:
 if j['input_mode']!='none':assert (ROOT/j['video_file']).is_file(), 'Missing video '+j['video_file']
if os.environ.get('VIDEOOOC_VALIDATE_ONLY')=='1':
 print(json.dumps({'requests_per_model':TARGET,'smoke_per_model':SMOKE_TARGET,'experiments':sorted(EXPERIMENTS),'all_video_paths_readable':True,'selected_models':ACTIVE_MODELS},ensure_ascii=False));sys.exit(0)
SECRET=os.environ.get('GEMINI_API_KEY' if args.provider=='google' else 'OPENAI_API_KEY') or getpass.getpass('Official API key (hidden): ')
if not SECRET or not SECRET.isascii():raise SystemExit('Missing valid-format stdin credential')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
config={'base_url':BASE,'models':MODELS,'requests_per_model':TARGET,'smoke_per_model':SMOKE_TARGET,'experiments':sorted(EXPERIMENTS),'confirmed_model_aliases':ALIASES,'prompts':PROMPTS,'manifest_sha256':sha(SRC/'data/inference_jobs.jsonl'),'source_sha256':sha(Path(__file__)),'gemini':{'input':'inlineData video/mp4','maxOutputTokens':8192,'thinking':'provider default, final separated','temperature':0},'gpt':{'input':'64 uniformly sampled JPEG frames, original resolution, quality85, detail auto','max_completion_tokens':2048,'temperature':'provider default'},'concurrency_per_model':2,'request_timeout_s':180,'retry_attempts':1,'gate':'all selected smoke requests have nonempty final, normal stop, confirmed returned model; successful smoke reused in full','billing':'optional user-supplied per-million rates; usage only is not a settlement receipt','pricing_schedule':PRICES,'created_date':'2026-11-25','adapter':'official_api_v1','provider':args.provider,'historical_result_equivalence':False}
cfg=OUT/'run_config.json'
if cfg.exists():assert json.loads(cfg.read_text())==config,'Configuration changed; use separate run'
else:cfg.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
(OUT/'input_jobs.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs))
write_lock=threading.Lock();media_locks=collections.defaultdict(threading.Lock);state_lock=threading.Lock();counts=collections.Counter();last_state={};stop_event=threading.Event()
def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def append(p,r):
 p.parent.mkdir(parents=True,exist_ok=True)
 with write_lock:
  with p.open('a') as f:f.write(json.dumps(r,ensure_ascii=False).replace(SECRET,'[REDACTED]')+'\n');f.flush();os.fsync(f.fileno())
def status(phase,**extra):
 with state_lock:
  last_state.update({'phase':phase,'updated':stamp(),'counts':dict(counts),**extra});p=OUT/'status.tmp';p.write_text(json.dumps(last_state,ensure_ascii=False,indent=2));p.replace(OUT/'status.json')
def media(j,frames=False):
 rel=j['video_file'];assert rel and not Path(rel).is_absolute() and '..' not in Path(rel).parts
 target=ROOT/rel;assert target.is_file(), 'Missing shared video: '+str(target)
 with media_locks[rel]:
  receipt={'video_sha256':sha(target),'video_bytes':target.stat().st_size}
  if not frames:return target,receipt
  folder=OUT/'frame_cache'/Path(rel).parent/Path(rel).stem;meta=folder/'metadata.json'
  if not meta.exists():
   import cv2
   cap=cv2.VideoCapture(str(target));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);assert n>0 and fps>0
   indices=[round(i*(n-1)/63) for i in range(64)];folder.mkdir(parents=True,exist_ok=True)
   for i,index in enumerate(indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,img=cap.read();assert ok,'Decode failed';assert cv2.imwrite(str(folder/f'{i:02}.jpg'),img,[cv2.IMWRITE_JPEG_QUALITY,85])
   cap.release();meta.write_text(json.dumps({'num_frames':64,'sampled_frame_indices':indices,'fps':fps,'original_frames':n,'resized':False}))
  return [folder/f'{i:02}.jpg' for i in range(64)],{**receipt,**json.loads(meta.read_text())}
def call(path,body):
 config='max-redirs = 0\nurl = '+json.dumps(BASE+path)+'\nheader = '+json.dumps(('x-goog-api-key: ' if args.provider=='google' else 'Authorization: Bearer ')+SECRET)+'\nheader = "Content-Type: application/json"\nrequest = "POST"\n'
 with tempfile.TemporaryDirectory(prefix='videoooc_request_') as tmp:
  p=Path(tmp)/'payload.json';p.write_text(json.dumps(body,ensure_ascii=False));config+='data-binary = '+json.dumps('@'+str(p))+'\n'
  r=subprocess.run(['curl','--compressed','-sS','--connect-timeout','15','--max-time','180','--write-out','\n%{http_code}','--config','-'],input=config,text=True,capture_output=True,timeout=190)
 raw,_,http=r.stdout.rpartition('\n')
 try:data=json.loads(raw)
 except Exception:data={'parse_error':True,'body_excerpt':raw[:1000]}
 return http,data,r.returncode,r.stderr[:500]
def cost(tag,usage):
 # A relay's prices and account ratios are deployment-specific. Never assume
 # historical rates apply to a different endpoint or date.
 rates=PRICES.get(tag)
 if rates is None:return None
 if args.provider=='google':
  prompt=usage.get('promptTokenCount',0);completion=usage.get('candidatesTokenCount',0)+usage.get('thoughtsTokenCount',0);cached=usage.get('cachedContentTokenCount',0)
 else:
  prompt=usage.get('prompt_tokens',0);completion=usage.get('completion_tokens',0);cached=usage.get('prompt_tokens_details',{}).get('cached_tokens',0)
 return (max(0,prompt-cached)*rates['input_per_million']+cached*rates.get('cached_input_per_million',rates['input_per_million'])+completion*rates['output_per_million'])/1e6
fatal=[]
def one(tag,j,phase):
 if stop_event.is_set():return False
 import base64
 start=time.perf_counter();prompt=PROMPTS[j['prompt_id']].format(question=j['question']);metadata={};visual=j['input_mode']!='none'
 try:
  if args.provider=='google':
   parts=[]
   if visual:
    v,metadata=media(j);parts.append({'inlineData':{'mimeType':'video/mp4','data':base64.b64encode(v.read_bytes()).decode()}})
   parts.append({'text':prompt});body={'contents':[{'role':'user','parts':parts}],'generationConfig':{'maxOutputTokens':8192,'temperature':0}};path='/v1beta/models/'+tag+':generateContent'
  else:
   parts=[]
   if visual:
    frames,metadata=media(j,True)
    parts=[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(f.read_bytes()).decode(),'detail':'auto'}} for f in frames]
   parts.append({'type':'text','text':prompt});body={'model':tag,'messages':[{'role':'user','content':parts}],'max_completion_tokens':2048,'stream':False};path='/v1/chat/completions'
  for attempt in range(1,2):
   t=time.perf_counter();http,data,exitcode,err=call(path,body);elapsed=time.perf_counter()-t
   if http=='200' and not data.get('error'):
    if args.provider=='google':
     candidate=(data.get('candidates')or[{}])[0];parts=candidate.get('content',{}).get('parts',[]);final=''.join(x.get('text','') for x in parts if not x.get('thought'));thinking=''.join(x.get('text','') for x in parts if x.get('thought')) or None;finish=candidate.get('finishReason');returned=data.get('modelVersion');usage=data.get('usageMetadata',{});normal=finish=='STOP'
    else:
     choice=(data.get('choices')or[{}])[0];msg=choice.get('message',{});final=msg.get('content') or '';thinking=msg.get('reasoning_content');finish=choice.get('finish_reason');returned=data.get('model');usage=data.get('usage',{});normal=finish=='stop'
    ok=bool(final.strip()) and normal and accepted_model(tag,returned)
    row={**j,'model_tag':tag,'model_version':returned,'prompt_text':prompt,'raw_answer':final,'final_answer':final,'thinking':thinking,'inference_status':'ok' if ok else 'invalid_or_truncated','finish_reason':finish,'latency_s':round(elapsed,4),'end_to_end_latency_s':round(time.perf_counter()-start,4),'run_time':stamp(),'attempt':attempt,'num_frames':64 if visual and args.provider!='google' else None,'video_grid_thw':None,'usage':usage,'estimated_cost_usd':cost(tag,usage),'api_response':data,'input_metadata':metadata,'source_phase':phase}
    append(OUT/'results'/j['exp']/tag/f"{j['prompt_id']}_{j['input_mode']}.jsonl",row)
    with state_lock:counts[tag]+=1
    status(phase);print(tag,j['sample_id'],j['variant'],j['prompt_id'],j['input_mode'],row['inference_status'],round(elapsed,2),flush=True)
    if not ok:stop_event.set();fatal.append({'model':tag,'key':key(j),'reason':'invalid_or_truncated_or_model_mismatch'})
    return ok
   append(OUT/'errors.jsonl',{'model':tag,**j,'attempt':attempt,'time':stamp(),'http':http,'curl_exit':exitcode,'error':err,'response':data,'latency_s':elapsed,'billing_uncertain':http=='000'})
   if http in ['401','403','400','404'] or attempt==1:
    stop_event.set();fatal.append({'model':tag,'key':key(j),'reason':'request_failed','http':http});return False
   time.sleep(min(30,5*attempt))
 except Exception as exc:
  append(OUT/'errors.jsonl',{'model':tag,**j,'time':stamp(),'exception':str(exc).replace(SECRET,'[REDACTED]')});stop_event.set();fatal.append({'model':tag,'key':key(j),'reason':'local_exception'});return False

def existing(tag):
 done=set()
 for p in (OUT/'results').glob('*/'+tag+'/*.jsonl'):
  for l in p.read_text().splitlines():
   r=json.loads(l);assert r['exp'] in EXPERIMENTS, 'Output directory contains another experiment selection'
   assert key(r) not in done, 'Duplicate result'
   assert key(r) in EXPECTED and all(r[k]==v for k,v in EXPECTED[key(r)].items()), 'Result/input mismatch'
   assert r['model_tag']==tag and r['prompt_text']==PROMPTS[r['prompt_id']].format(question=r['question']), 'Model/prompt mismatch'
   assert r['final_answer'].strip() and r.get('finish_reason') in {'STOP','stop'}, 'Existing failed/truncated record requires review'
   assert accepted_model(tag,r['model_version']), 'Returned model ID requires provider confirmation; keep record, do not silently regenerate'
   assert r['inference_status']=='ok' or (r['inference_status']=='invalid_or_truncated' and r['model_version'] in ALIASES.get(tag,[])), 'Unreviewed result'
   done.add(key(r))
 counts[tag]=len(done);return done

def lane(tag,items,phase):
 done=existing(tag);pending=[j for j in items if key(j) not in done]
 with cf.ThreadPoolExecutor(max_workers=2) as pool:
  it=iter(pending);active=set()
  for _ in range(2):
   j=next(it,None)
   if j:active.add(pool.submit(one,tag,j,phase))
  while active:
   ready,active=cf.wait(active,return_when=cf.FIRST_COMPLETED)
   for f in ready:
    f.result()
    if not stop_event.is_set():
     j=next(it,None)
     if j:active.add(pool.submit(one,tag,j,phase))
try:
 for tag in ACTIVE_MODELS:existing(tag)
 status('smoke_running',target_per_model=TARGET,active_models=ACTIVE_MODELS)
 with cf.ThreadPoolExecutor(max_workers=2) as p:list(p.map(lambda tag:lane(tag,smoke,'smoke_running'),ACTIVE_MODELS))
 passed=not stop_event.is_set() and all({key(j) for j in smoke}<=existing(tag) for tag in ACTIVE_MODELS)
 (OUT/'smoke_validation.json').write_text(json.dumps({'passed':passed,'expected_per_model':SMOKE_TARGET,'models':ACTIVE_MODELS,'fatal':fatal,'time':stamp()},ensure_ascii=False,indent=2))
 if passed and STAGE=='pipeline':
  status('full_running',smoke_passed=True)
  with cf.ThreadPoolExecutor(max_workers=2) as p:list(p.map(lambda tag:lane(tag,jobs,'full_running'),ACTIVE_MODELS))
  complete=not stop_event.is_set() and all(len(existing(tag))==TARGET for tag in ACTIVE_MODELS)
  status('complete' if complete else ('blocked' if stop_event.is_set() else 'selected_models_complete'),fatal=fatal,active_models=ACTIVE_MODELS)
 else:status('smoke_complete' if passed else 'blocked',smoke_passed=passed,fatal=fatal)
except Exception as exc:status('blocked',fatal=fatal,exception=str(exc).replace(SECRET,'[REDACTED]'));raise
finally:SECRET=''
