"""Same-env BASE/2tier/3tier MAIN P0/P1 evaluation, one video/question per call.
Inference only; no paid judge. Never deletes empty answers or source attempts.
Prepared while training waits; actual GPU evaluation is not yet validated.
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['TOKENIZERS_PARALLELISM']='false'
import pathlib,argparse,json,hashlib,time,fcntl,subprocess,traceback,math
P=pathlib.Path(__file__).parent
PROMPTS={
 'P0':'Answer the following question based only on the video.\nQuestion: {question}\nGive a brief and direct answer.',
 'P1':'Answer the question using only evidence visible in the video.\nIf the evidence is sufficient, answer directly.\nIf the evidence supports only a plausible guess, state the guess with clear uncertainty.\nIf the video does not provide enough evidence, answer: "Cannot be determined from the video."\nQuestion: {question}'}
ap=argparse.ArgumentParser();ap.add_argument('--condition',choices=['base','binary','ternary'],required=True)
ap.add_argument('--split',choices=['main','dev'],default='main')
ap.add_argument('--model-path',required=True);ap.add_argument('--adapter-path')
ap.add_argument('--dataset-root',required=True);ap.add_argument('--input',default=str(P/'data/main_eval_1721.jsonl'))
ap.add_argument('--output-dir',required=True);ap.add_argument('--gpu',type=int,required=True)
ap.add_argument('--check-inputs',action='store_true')
ap.add_argument('--validation-ids',nargs='+',help='Run these existing IDs only; preserve successful requests for the full run.')
ap.add_argument('--gpu-lock-fd',type=int,help=argparse.SUPPRESS)
a=ap.parse_args()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
rows=[json.loads(l) for l in pathlib.Path(a.input).read_text(encoding='utf-8').splitlines() if l.strip()]
assert len(rows)==len({r['sample_id'] for r in rows})==({'main':1721,'dev':62}[a.split])
assert all(set(r)=={'sample_id','video_id','question_zh','video_relative_path'} for r in rows),'Evaluation manifest must not carry labels/reference/grounding'
if a.validation_ids:assert set(a.validation_ids).issubset({r['sample_id'] for r in rows})
missing=[]
for r in rows:
 path=pathlib.PurePosixPath(r['video_relative_path'])
 assert not path.is_absolute() and '..' not in path.parts
 if a.split=='main':
  assert len(path.parts)==2 and path.parts[0] in ['Charades_v1_480_mute','Ego4D_v2_540ss_mute']
 else:
  assert len(path.parts)==1 and path.name==r['video_id']+'.mp4'
 if not (pathlib.Path(a.dataset_root)/r['video_relative_path']).is_file():missing.append(r['video_relative_path'])
if a.check_inputs:
 print(json.dumps({'manifest_rows':len(rows),'expected_requests':2*len(rows),'missing_video_paths':sorted(set(missing)),'reference_label_in_model_input':False,'gpu_used':False},ensure_ascii=False));raise SystemExit(2 if missing else 0)
assert not missing,f'{len(set(missing))} MAIN videos unavailable; no evaluation started'
if a.condition=='base':assert not a.adapter_path
else:assert a.adapter_path and (pathlib.Path(a.adapter_path)/'adapter_model.safetensors').is_file()
out=pathlib.Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
lock=(out/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def idle():
 s=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 free,util=map(int,s.strip().split(','));return free>=23500 and util<=3
os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
import torch,transformers
from transformers import AutoModelForImageTextToText,AutoProcessor,set_seed
set_seed(20261102);torch.set_num_threads(2)
signature={'condition':a.condition,'split':a.split,'manifest_sha256':sha(a.input),'code_sha256':sha(__file__),'prompts':PROMPTS,'max_new_tokens':128,'do_sample':False,'precision':'bf16 model + bf16 autocast','processor':'original defaults + cap_pixels_per_frame=True','model_path':str(pathlib.Path(a.model_path).resolve()),'adapter_sha256':sha(pathlib.Path(a.adapter_path)/'adapter_model.safetensors') if a.adapter_path else None,'adapter_config_sha256':sha(pathlib.Path(a.adapter_path)/'adapter_config.json') if a.adapter_path else None,'torch':torch.__version__,'transformers':transformers.__version__,'dataset_root':str(pathlib.Path(a.dataset_root).resolve())}
fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
sigfile=out/'run_signature.json'
if sigfile.exists():assert json.loads(sigfile.read_text())==signature,'Different protocol/input; use new output directory'
else:sigfile.write_text(json.dumps(signature,ensure_ascii=False,indent=2),encoding='utf-8')
expected={f"{a.condition}__{r['sample_id']}__{pr}" for r in rows for pr in PROMPTS}
raw=out/'raw_predictions.jsonl';success={};attempts={}
if raw.exists():
 for l in raw.read_text(encoding='utf-8').splitlines():
  if not l.strip():continue
  r=json.loads(l);assert r['protocol_sha256']==fingerprint and r['request_id'] in expected
  attempts[r['request_id']]=attempts.get(r['request_id'],0)+1
  if r['inference_status']=='ok':
   assert r['request_id'] not in success,'Duplicate success'
   assert isinstance(r['raw_answer'],str) and math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds']>0
   success[r['request_id']]=r
def status(state):
 (out/'status.json').write_text(json.dumps({'state':state,'condition':a.condition,'completed':len(success),'expected':len(expected),'empty_outputs':sum(not r['raw_answer'].strip() for r in success.values()),'time':time.time()},indent=2))
if set(success)==expected:
 status('complete_inference_only_scoring_pending');raise SystemExit(0)
gpu_lock_path=pathlib.Path.home()/f'.videoooc_gpu_{a.gpu}.lock'
if a.gpu_lock_fd is None:
 gpu_lock=gpu_lock_path.open('a')
else:
 assert os.fstat(a.gpu_lock_fd).st_ino==gpu_lock_path.stat().st_ino
 assert os.fstat(a.gpu_lock_fd).st_dev==gpu_lock_path.stat().st_dev
 gpu_lock=os.fdopen(os.dup(a.gpu_lock_fd),'a')
fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert idle(),'GPU is not empty';time.sleep(10);assert idle(),'GPU became busy'
processor=AutoProcessor.from_pretrained(a.model_path,trust_remote_code=True)
processor.save_pretrained(out/'processor_snapshot')
model=AutoModelForImageTextToText.from_pretrained(a.model_path,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda:0')
if a.adapter_path:
 from peft import PeftModel
 model=PeftModel.from_pretrained(model,a.adapter_path,is_trainable=False)
model.eval();model.config.use_cache=True
eos=model.generation_config.eos_token_id
eos_ids=set(eos if isinstance(eos,list) else [eos])
for row in rows:
 if a.validation_ids and row['sample_id'] not in a.validation_ids:continue
 for prompt_id,template in PROMPTS.items():
  key=f"{a.condition}__{row['sample_id']}__{prompt_id}"
  if key in success:continue
  while attempts.get(key,0)<3:
   record={'request_id':key,'sample_id':row['sample_id'],'video_id':row['video_id'],'question_zh':row['question_zh'],'video_relative_path':row['video_relative_path'],'condition':a.condition,'prompt_id':prompt_id,'protocol_sha256':fingerprint,'attempt':attempts.get(key,0)+1,'gpu_physical_index':a.gpu}
   inputs=None;generated=None;tail=None
   try:
    torch.cuda.synchronize();start=time.perf_counter()
    video=str(pathlib.Path(a.dataset_root)/row['video_relative_path'])
    messages=[{'role':'user','content':[{'type':'video','video':video},{'type':'text','text':template.format(question=row['question_zh'])}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[text],videos=[video],return_tensors='pt',cap_pixels_per_frame=True)
    n=inputs['input_ids'].shape[-1];grid=inputs['video_grid_thw'].tolist()
    inputs={k:v.to('cuda:0') if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
    with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
     generated=model.generate(**inputs,max_new_tokens=128,do_sample=False,use_cache=True)
    tail=generated[0,n:];answer=processor.tokenizer.decode(tail,skip_special_tokens=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    end_eos=bool(tail.numel() and int(tail[-1]) in eos_ids)
    record.update(inference_status='ok',raw_answer=answer,response_status='invalid_empty' if not answer.strip() else 'nonempty',elapsed_seconds=elapsed,input_tokens=n,output_tokens=int(tail.numel()),video_grid_thw=grid,finish_reason='eos' if end_eos else ('length' if tail.numel()>=128 else 'other'))
    assert math.isfinite(elapsed) and elapsed>0
   except Exception as error:
    record.update(inference_status='error',error=repr(error),traceback=traceback.format_exc())
   finally:
    attempts[key]=attempts.get(key,0)+1
    with raw.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
    del inputs,generated,tail
   if record['inference_status']=='ok':success[key]=record;break
   torch.cuda.empty_cache()
   # Deterministic resource/kernel errors need diagnosis; do not loop OOM.
   if 'out of memory' in record['error'].lower() or 'device-side assert' in record['error'].lower():
    status('failed_requires_diagnosis');raise RuntimeError(record['error'])
  status('running')
  if key not in success:status('failed_requires_diagnosis');raise RuntimeError('Request failed three times: '+key)
if a.validation_ids:
 assert all(f"{a.condition}__{sample}__{prompt}" in success for sample in a.validation_ids for prompt in PROMPTS)
 status('validation_subset_complete_full_evaluation_pending');raise SystemExit(0)
assert set(success)==expected
status('complete_inference_only_scoring_pending')
