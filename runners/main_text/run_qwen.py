"""Video-OOC v11. One authenticated local Qwen checkpoint, experiment and prompt per process."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import time
import traceback

PROMPTS = {
    "P0": "Answer the following question based only on the video.\nQuestion: {question}\nGive a brief and direct answer.",
    "P1": 'Answer the question using only evidence visible in the video.\nIf the evidence is sufficient, answer directly.\nIf the evidence supports only a plausible guess, state the guess with clear uncertainty.\nIf the video does not provide enough evidence, answer: "Cannot be determined from the video."\nQuestion: {question}',
}
MODELS = {
    "Qwen3-VL-32B-Instruct": ("0cfaf48183f594c314753d30a4c4974bc75f3ccb", 128),
    "Qwen3-VL-8B-Thinking": ("92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b", 2048),
}
GIB = 1024 ** 3
VERSION = "VideoOOC_v11_Qwen_MAIN_TEXT_20261104_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gpu_allocation(visible, indices, count):
    devices = (visible or "").split(",")
    uuid = r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
    numeric = all(d.isdecimal() for d in devices)
    require((numeric or all(re.fullmatch(uuid,d) for d in devices)) and len(set(devices))==len(devices)==count,
            "Scheduler must expose its complete locked physical GPU index or UUID list")
    physical = (indices or (visible if numeric else "")).split(",")
    require(len(set(physical))==len(physical)==count and all(d.isdecimal() for d in physical),
            "Scheduler must provide VIDEO_OOC_GPU_INDICES for UUID-based visibility")
    if numeric:
        require(physical==devices,"Inconsistent physical GPU indices")
    return devices,[int(i) for i in physical]


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def append_row(path, record):
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def manifest_rows(path, media_root, video):
    with path.open(encoding="utf-8-sig") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    required = {"sample_id", "video_id", "video_relative_path", "question_zh"}
    require(len(rows) == 1721, "Expected the approved 1721-question original manifest")
    ids = set()
    paths = {}
    for r in rows:
        require(set(r) == required, "Questions manifest must contain only the four approved input fields")
        sid = r["sample_id"]
        require(isinstance(sid,str) and re.fullmatch(r"p23_\d+",sid) and sid not in ids, "Invalid or duplicate question ID")
        ids.add(sid)
        require(isinstance(r["question_zh"], str) and r["question_zh"].strip(), f"Missing question {sid}")
        rel = PurePosixPath(r["video_relative_path"])
        require(len(rel.parts)==2 and rel.parts[0] in ("Charades_v1_480_mute", "Ego4D_v2_540ss_mute")
                and rel.name == r["video_id"]+".mp4" and "\\" not in str(rel), f"Invalid media mapping {sid}")
        require(paths.setdefault(r["video_id"], str(rel)) == str(rel), f"Mixed media associations {sid}")
    require(len(paths)==598, "Expected 598 distinct MAIN videos")
    if video:
        require(media_root is not None and media_root.is_dir(), "MAIN requires --dataset-root with the mute video folders")
        for rel in paths.values():
            require((media_root / rel).is_file(), f"Missing MAIN video {rel}")
    return rows


def model_identity(root, tag, revision):
    root = root.resolve()
    wanted = MODELS[tag][0]
    require(revision == wanted, f"Expected revision {wanted}")
    require(root.name == wanted and root.parent.name == "snapshots" and root.parent.parent.name == "models--Qwen--"+tag,
            f"Use the verified Qwen/{tag} HF snapshot, not a loose/instruct/thinking substitute")
    config = read_json(root/"config.json")
    require(config.get("model_type")=="qwen3_vl" and "Qwen3VLForConditionalGeneration" in config["architectures"]
            and not config.get("quantization_config"), "Unexpected or quantized model architecture")
    index = read_json(root/"model.safetensors.index.json")
    shard_names = sorted(set(index["weight_map"].values()))
    require(shard_names, "No model shards")
    shard_sizes = {}
    for name in shard_names:
        require(name==Path(name).name and name.endswith(".safetensors"), "Invalid shard name")
        size = (root/name).stat().st_size
        require(size>0, f"Empty model shard {name}")
        shard_sizes[name] = size
    config_files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in (".json",".jinja",".txt"))
    return {"repo_id":"Qwen/"+tag, "revision":revision,
            "config_files_sha256":{p.name:file_hash(p) for p in config_files},
            "weights_bytes":shard_sizes}


def split_decoded(raw, thinking_model, hit_limit):
    if thinking_model:
        reasoning, close, final = raw.rpartition("</think>")
        if not close:
            return raw.removeprefix("<think>").strip(), "", "truncated", "invalid_missing_think_close"
        reasoning = reasoning.removeprefix("<think>").strip()
        final = final.strip()
    else:
        reasoning, final = None, raw
    if hit_limit:
        return reasoning, final, "truncated", "invalid_length"
    return reasoning, final, "ok", "valid" if final.strip() else "invalid_empty"


def frame_count(metadata):
    require(isinstance(metadata, (list,tuple)) and len(metadata)==1, "Expected metadata for one source video")
    entry = metadata[0]
    indices = entry.get("frames_indices") if isinstance(entry,dict) else getattr(entry,"frames_indices",None)
    require(indices is not None and len(indices)>0, "Processor did not report the actual sampled frame indices")
    return [int(x) for x in indices]


def request_id(tag, exp, prompt, row):
    return "|".join((exp, tag, prompt, row["sample_id"]))


def validate_done(r, source, args, signature_hash):
    sid = r["sample_id"]
    require(r["protocol_sha256"]==signature_hash and r["request_id"]==request_id(args.model_tag,args.exp,args.prompt,source),
            f"Unexpected protocol or request ID for {sid}")
    require(r["model_tag"]==args.model_tag and r["prompt_id"]==args.prompt and r["exp"]==args.exp and
            r["video_id"]==source["video_id"] and r["question"]==source["question_zh"] and
            r["prompt_text"]==PROMPTS[args.prompt].format(question=source["question_zh"]), f"Source mismatch {sid}")
    video = args.exp=="MAIN"
    require(r["input_mode"]==("video" if video else "none") and
            r["video_file"]==(source["video_relative_path"] if video else None), f"Modality/source mismatch {sid}")
    require((isinstance(r["num_frames"],int) and r["num_frames"]>0) if video else r["num_frames"] is None, f"Frame count mismatch {sid}")
    require(isinstance(r["raw_answer"],str) and math.isfinite(r["elapsed_seconds"]) and r["elapsed_seconds"]>0 and
            r["elapsed_seconds"]==r["latency_s"], f"Invalid raw or latency {sid}")
    decoded = split_decoded(r["raw_answer"],args.model_tag.endswith("Thinking"),r["finish_reason"]=="length")
    require(decoded==(r["thinking"],r["final_answer"],r["inference_status"],r["response_status"]),f"Invalid parsed record {sid}")


def environment():
    packages = {}
    for name in ("torch","transformers","accelerate","torchcodec","torchvision","tokenizers"):
        try:
            packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name]=None
    return packages


def generate_one(row, args, processor, model, torch, input_device, eos_ids):
    video = args.exp=="MAIN"
    is_thinking = args.model_tag.endswith("Thinking")
    prompt = PROMPTS[args.prompt].format(question=row["question_zh"])
    for gpu in range(torch.cuda.device_count()):
        torch.cuda.synchronize(gpu)
    start=time.perf_counter()
    content = [{"type":"text","text":prompt}]
    if video:
        path=str(args.dataset_root/row["video_relative_path"])
        content.insert(0,{"type":"video","video":path})
    rendered=processor.apply_chat_template([{"role":"user","content":content}], tokenize=False,add_generation_prompt=True)
    require(("<|video_pad|>" in rendered)==video, "Template modality mismatch")
    if is_thinking:
        require(rendered.rstrip().endswith("<think>"), "Thinking assistant generation prefix missing")
    if video:
        # Request the observational sampled-frame metadata; preserve the original processor defaults.
        inputs=processor(text=[rendered],videos=[path],return_tensors="pt",cap_pixels_per_frame=True,return_metadata=True)
        sampled=frame_count(inputs.pop("video_metadata",None))
        require(inputs.get("pixel_values_videos") is not None and inputs.get("video_grid_thw") is not None,
                "Actual MAIN pixel tensors missing")
        grid=inputs["video_grid_thw"].tolist()
    else:
        inputs=processor(text=[rendered],return_tensors="pt")
        require(all(inputs.get(k) is None for k in ("pixel_values", "pixel_values_videos","video_grid_thw","image_grid_thw")),
                "Unexpected visual input in TEXT")
        sampled,grid=None,None
    n=int(inputs["input_ids"].shape[-1])
    tensors={k:v.to(input_device) for k,v in inputs.items()}
    with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        generated=model.generate(**tensors,max_new_tokens=MODELS[args.model_tag][1],do_sample=False,num_beams=1,use_cache=True)
    token_ids=generated[0,n:].tolist()
    eos=bool(token_ids and token_ids[-1] in eos_ids)
    reason="eos" if eos else ("length" if len(token_ids)>=MODELS[args.model_tag][1] else "other")
    # The closing think token may be registered as a special token. Retain it and remove only the transport EOS.
    raw=processor.tokenizer.decode(token_ids[:-1] if eos else token_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    thinking,final,status,response=split_decoded(raw,is_thinking,reason=="length")
    for gpu in range(torch.cuda.device_count()):
        torch.cuda.synchronize(gpu)
    elapsed=time.perf_counter()-start
    return dict(request_id=request_id(args.model_tag,args.exp,args.prompt,row),exp=args.exp,
                sample_id=row["sample_id"],item=None,variant="orig",model_tag=args.model_tag,
                model_version=f"Qwen/{args.model_tag}@{args.model_revision}",prompt_id=args.prompt,
                input_mode="video" if video else "none",video_file=row["video_relative_path"] if video else None,
                video_id=row["video_id"],question=row["question_zh"],prompt_text=prompt,raw_answer=raw,
                final_answer=final,thinking=thinking,inference_status=status,response_status=response,
                num_frames=len(sampled) if video else None,sampled_frame_indices=sampled,video_grid_thw=grid,
                latency_s=elapsed,elapsed_seconds=elapsed,run_time=utc(),input_tokens=n,output_tokens=len(token_ids),
                generated_token_ids=token_ids,finish_reason=reason,max_new_tokens=MODELS[args.model_tag][1],
                gpu_ids=os.environ["CUDA_VISIBLE_DEVICES"],physical_gpu_indices=args.physical_gpu_indices,protocol_version=VERSION)


def run(args):
    import fcntl
    require(args.exp=="MAIN" or args.prompt=="P0", "Only TEXT P0 was authorized")
    require(args.limit is None or args.limit>0, "--limit must be positive")
    rows=manifest_rows(args.input,args.dataset_root,args.exp=="MAIN")
    by_id={r["sample_id"]:r for r in rows}
    if args.only_ids:
        require(set(args.only_ids)<=set(by_id) and len(set(args.only_ids))==len(args.only_ids),"Unknown or duplicate --only-ids")
    identity=model_identity(args.model_path,args.model_tag,args.model_revision)
    out=args.output_root/"results"/args.exp/args.model_tag
    out.mkdir(parents=True,exist_ok=True)
    mode="video" if args.exp=="MAIN" else "none"
    prefix=out/f"{args.prompt}_{mode}"
    raw=prefix.with_suffix(".jsonl")
    status_path=prefix.with_suffix(".status.json")
    sigfile=prefix.with_suffix(".signature.json")
    with prefix.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        from transformers import AutoModelForImageTextToText,AutoProcessor,set_seed
        gpu_ids,args.physical_gpu_indices=gpu_allocation(os.environ.get("CUDA_VISIBLE_DEVICES"),os.environ.get("VIDEO_OOC_GPU_INDICES"),torch.cuda.device_count())
        models=[torch.cuda.get_device_name(i) for i in range(len(gpu_ids))]
        require(all(torch.cuda.get_device_capability(i)[0]>=8 for i in range(len(gpu_ids))),"Native bf16 GPUs required")
        signature=dict(protocol=VERSION,model=identity,manifest_sha256=file_hash(args.input),code_sha256=file_hash(Path(__file__)),
                       model_tag=args.model_tag,exp=args.exp,prompt_id=args.prompt,prompt_template=PROMPTS[args.prompt],
                       max_new_tokens=MODELS[args.model_tag][1],do_sample=False,num_beams=1,dtype="bf16",attention="sdpa",
                       processor="native Qwen; MAIN cap_pixels_per_frame=True and original video sampling; TEXT has no visual input",
                       packages=environment(),gpu_models=models,sharding="balanced_low_0" if len(models)>1 else "single",
                       reserve_first_gib=3.5,reserve_other_gib=2.0,allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                       timing="synchronized preprocessing+generation+decode; excludes loading, serialization and disk I/O")
        signature_hash=hashlib.sha256(json.dumps(signature,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if sigfile.exists():
            require(read_json(sigfile)==signature,"Signature changed. Use another run/output directory.")
        else:
            require(not raw.exists(),"Raw file without its signature")
            atomic_json(sigfile,signature)
        done=set()
        counts=Counter()
        if raw.exists():
            with raw.open(encoding="utf-8") as f:
                for line in f:
                    record=json.loads(line)
                    sid=record["sample_id"]
                    require(sid in by_id and sid not in done,f"Unexpected/duplicate record {sid}")
                    validate_done(record,by_id[sid],args,signature_hash)
                    done.add(sid)
                    counts[record["response_status"]]+=1
        def status(state,**more):
            atomic_json(status_path,dict(state=state,completed=len(done),expected=len(rows),counts=counts,updated_at=utc(),**more))
        selected=set(args.only_ids or by_id)
        pending=[r for r in rows if r["sample_id"] in selected and r["sample_id"] not in done][:args.limit]
        if not pending:
            status("complete" if len(done)==len(rows) else "subset_complete")
            return
        try:
            status("loading")
            set_seed(20261104)
            torch.set_num_threads(2)
            processor=AutoProcessor.from_pretrained(args.model_path,local_files_only=True)
            kwargs={}
            device_map="cuda:0"
            if len(models)>1:
                device_map="balanced_low_0"
                budgets={i:int(torch.cuda.mem_get_info(i)[0]-(3.5 if i==0 else 2.0)*GIB) for i in range(len(models))}
                require(min(budgets.values())>0 and sum(budgets.values())>sum(identity["weights_bytes"].values()),
                        "Not enough GPU weight budget after activation reserve")
                kwargs["max_memory"]={**budgets,"cpu":0}
            model=AutoModelForImageTextToText.from_pretrained(args.model_path,local_files_only=True,dtype=torch.bfloat16,
                        attn_implementation="sdpa",device_map=device_map,**kwargs).eval()
            require(not getattr(model,"is_quantized",False),"Quantized model forbidden")
            require(all(p.is_cuda for p in model.parameters()),"CPU/disk weight offload is not the protocol")
            require(all(p.dtype==torch.bfloat16 for p in model.parameters() if p.is_floating_point()),"Unexpected weight dtype")
            input_device=model.get_input_embeddings().weight.device
            eos=model.generation_config.eos_token_id
            eos_ids=set(eos if isinstance(eos,list) else [eos])
            parameter_devices=Counter(str(p.device) for p in model.parameters())
            status("running",parameter_devices=dict(parameter_devices),
                   dispatch_map={k:str(v) for k,v in (getattr(model,"hf_device_map",None) or {}).items()})
            for row in pending:
                sid=row["sample_id"]
                t=time.perf_counter()
                try:
                    rec=generate_one(row,args,processor,model,torch,input_device,eos_ids)
                    rec["protocol_sha256"]=signature_hash
                    validate_done(rec,row,args,signature_hash)
                except Exception as err:
                    append_row(prefix.with_suffix(".errors.jsonl"),dict(request_id=request_id(args.model_tag,args.exp,args.prompt,row),
                           sample_id=sid,inference_status="error",raw_answer=repr(err),elapsed_seconds=time.perf_counter()-t,
                           traceback=traceback.format_exc(),protocol_sha256=signature_hash,run_time=utc()))
                    raise
                append_row(raw,rec)
                done.add(sid)
                counts[rec["response_status"]]+=1
                if len(done)%10==0 or sid==pending[-1]["sample_id"]:
                    status("running")
                    print(f"{args.exp}/{args.model_tag}/{args.prompt} {len(done)}/{len(rows)} {dict(counts)}",flush=True)
            status("complete" if len(done)==len(rows) else "subset_complete")
        except Exception as err:
            status("failed_requires_diagnosis",error=repr(err),traceback=traceback.format_exc())
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-tag",required=True,choices=MODELS)
    p.add_argument("--model-path",type=Path,required=True)
    p.add_argument("--model-revision",required=True)
    p.add_argument("--exp",choices=("MAIN","TEXT"),required=True)
    p.add_argument("--prompt",choices=PROMPTS,required=True)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--dataset-root",type=Path)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--only-ids",nargs="+")
    p.add_argument("--limit",type=int)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var,"2")
    run(p.parse_args())


if __name__=="__main__":
    main()

