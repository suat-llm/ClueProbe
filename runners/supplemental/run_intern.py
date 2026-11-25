"""INDUCE adapter retaining the recorded MAIN model/frame/generation settings."""
import argparse,fcntl,hashlib,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',choices=['internvl','internvideo'],required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--model-path',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--media-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
D=a.output_root.resolve();D.mkdir(parents=True,exist_ok=True)
lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
import torch
from transformers import AutoModel,AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from video_frames import load_video
P={'P0':'Answer the following question based only on the video.\nQuestion: {question}\nGive a brief and direct answer.','P1':'Answer the question using only evidence visible in the video.\nIf the evidence is sufficient, answer directly.\nIf the evidence supports only a plausible guess, state the guess with clear uncertainty.\nIf the video does not provide enough evidence, answer: "Cannot be determined from the video."\nQuestion: {question}'}
tag,folder,nframes=('InternVL3.5-8B','InternVL3_5-8B',32) if a.model=='internvl' else ('InternVideo2.5-Chat-8B','InternVideo2_5_Chat_8B',128)
mp=str(a.model_path)
kw=dict(trust_remote_code=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=False)
if a.model=='internvideo':
 assert torch.cuda.device_count()==2
 kw.update(device_map='auto',max_memory={0:'24GiB',1:'24GiB'})
m=AutoModel.from_pretrained(mp,**kw).eval()
if a.model=='internvl':m=m.to('cuda:0');m.language_model.set_attn_implementation('sdpa')
tok=AutoTokenizer.from_pretrained(mp,trust_remote_code=True,**({'use_fast':False} if a.model=='internvl' else {}))
gen={'max_new_tokens':128,'do_sample':False}
if a.model=='internvideo':gen.update(temperature=0.0,top_p=0.1,num_beams=1)
jobs=[json.loads(x) for x in a.manifest.read_text().splitlines()];assert len(jobs)==6794
from video_ooc.manifest import validate_jobs
validate_jobs(jobs,a.media_root)
if a.smoke:
 jobs=[next(x for x in jobs if x['input_mode']==mode and x['prompt_id']==pid) for mode,pid in [('video','P0'),('video','P1'),('none','P1')]]
else:
 first=[next(x for x in jobs if x['input_mode']==mode and x['prompt_id']==pid) for mode,pid in [('video','P0'),('video','P1'),('none','P1')]]
 jobs=first+[x for x in jobs if x not in first]
result=D/('smoke.jsonl' if a.smoke else 'predictions.jsonl')
assert not result.exists(),'Explicit recovery required; do not overwrite existing answers'
config=dict(model=tag,num_frames=nframes,input_size=448,max_tiles=1,max_new_tokens=128,greedy=True,main_settings_source='recorded Chinese MAIN 20261025',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,prompts=P,source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(D/('smoke_config.json' if a.smoke else 'run_config.json')).write_text(json.dumps(config,indent=2))
vision_device=next(m.vision_model.parameters()).device
capture={}
original_generate=m.language_model.generate
def audited_generate(*args,**kwargs):
 out=original_generate(*args,**kwargs)
 seq=out.sequences if hasattr(out,'sequences') else out
 ids=seq[0].tolist();capture['ids']=ids
 return out
m.language_model.generate=audited_generate
for i,j in enumerate(jobs):
 capture.clear()
 prompt=P[j['prompt_id']].format(question=j['question']);start=time.monotonic()
 pixels=None;patches=[];prefix=''
 if j['input_mode']!='none':
  path=a.media_root/j['video_file'];assert path.is_file()
  pixels,patches=load_video(str(path),input_size=448,num_segments=nframes)
  assert len(patches)==nframes
  pixels=pixels.to(device=vision_device,dtype=torch.bfloat16)
  prefix=''.join(f'Frame{k+1}: <image>\n' for k in range(len(patches)))
 with torch.inference_mode():answer=m.chat(tok,pixels,prefix+prompt,gen.copy(),num_patches_list=patches,history=None,return_history=False)
 assert isinstance(answer,str) and answer.strip(),'Empty answer; stop'
 ids=capture.get('ids',[]);eos=m.language_model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
 terminated=bool(ids) and ids[-1] in eos
 row={**j,'model_tag':tag,'final_answer':answer,'raw_answer':answer,'prompt_text':prompt,'inference_status':'ok' if terminated else 'technical_invalid','num_frames':nframes if pixels is not None else 0,'latency_s':time.monotonic()-start,'generation_settings':gen,'generated_token_ids':ids,'output_tokens':len(ids),'finish_reason':'eos' if terminated else 'length_or_unknown'}
 with result.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
 assert terminated,'Generation termination could not be verified; saved row, stopped'
 if i==2 and not a.smoke:(D/'automatic_smoke_acceptance.json').write_text(json.dumps({'passed':True,'rows':3,'automatic_technical_checks':True,'checks':['nonempty','frames','fresh_context','generation_termination']}))
 (D/'status.json').write_text(json.dumps(dict(state='smoke_running' if a.smoke else 'running',completed=i+1,expected=len(jobs))))
(D/'status.json').write_text(json.dumps(dict(state='awaiting_smoke_review' if a.smoke else 'complete',completed=len(jobs))))
