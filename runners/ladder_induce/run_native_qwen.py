"""Native-processor Qwen8B adapters aligned to the received MAIN evidence.

This deliberately does not use the Qwen2B qwen-vl-utils/decord preprocessing.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata as md
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from run_qwen2b import PROMPTS, key, read_rows, validate, smoke_jobs

REVISIONS = {
    "Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "Qwen3-VL-8B-Thinking": "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b",
}


def split(raw, thinking, limited):
    if thinking:
        reasoning, close, final = raw.rpartition("</think>")
        if not close:
            return raw.removeprefix("<think>").strip(), "", "truncated"
        reasoning = reasoning.removeprefix("<think>").strip()
        final = final.strip()
    else:
        reasoning, final = None, raw
    return reasoning, final, "truncated" if limited else "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", choices=REVISIONS, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    tag = args.model_tag
    thinking = tag.endswith("Thinking")
    limit = 2048 if thinking else 128
    manifest = root / "data/inference_jobs.jsonl"
    jobs = read_rows(manifest)
    validate(jobs, root)
    if args.smoke:
        jobs = smoke_jobs(jobs)
    assert md.version("transformers") == "5.16.1", "Use the dedicated MAIN-compatible environment"
    if thinking:
        assert args.model_path.name == REVISIONS[tag] and args.model_path.parent.name == "snapshots"
    else:
        receipt = json.loads((root / "data/qwen8b_identity_validation.json").read_text())
        assert receipt["passed"] and receipt["model_path"] == str(args.model_path)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # Preserve the explicit Instruct MAIN decoder patch; Thinking was native.
    if not thinking:
        import transformers.video_processing_utils as vpu
        original_loader = vpu.load_video
        def pyav_loader(video_url_or_urls, backend=None, **kwargs):
            return original_loader(video_url_or_urls, backend="pyav", **kwargs)
        vpu.load_video = pyav_loader

    output = root / ("smoke" if args.smoke else "results")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / f"{tag}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_dir = root / ("smoke_configs" if args.smoke else "configs")
    config_dir.mkdir(exist_ok=True)
    config = {
        "model_tag": tag, "model_path": str(args.model_path),
        "model_version": f"Qwen/{tag}@{REVISIONS[tag]}",
        "revision_evidence": "pinned snapshot" if thinking else "current 13-file SHA equivalence, not historical execution commit attestation",
        "dtype": "bfloat16", "generation": {"do_sample": False, "max_new_tokens": limit},
        "video_processor": "native, cap_pixels_per_frame=True; no sampling/resolution overrides",
        "decoder": "native" if thinking else "forced_pyav",
        "prompts": PROMPTS, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "versions": {p: md.version(p) for p in ["torch", "transformers", "torchvision", "tokenizers", "av"]},
        "gpu": torch.cuda.get_device_name(0), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host": os.uname().nodename,
    }
    config_file = config_dir / f"run_config_{tag}.json"
    if config_file.exists():
        assert json.loads(config_file.read_text()) == config, "Cannot mix configurations"
    else:
        config_file.write_text(json.dumps(config, indent=2) + "\n")
    done = set()
    for p in output.glob(f"*/{tag}/*.jsonl"):
        for r in read_rows(p):
            assert key(r) not in done and r["inference_status"] in {"ok", "truncated"}
            done.add(key(r))
    if all(key(j) in done for j in jobs):
        print("Already complete", flush=True)
        return
    kwargs = dict(dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    if thinking:
        kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, **kwargs).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for i, job in enumerate(jobs, 1):
        if key(job) in done:
            continue
        prompt = PROMPTS[job["prompt_id"]].format(question=job["question"])
        video = job["input_mode"] != "none"
        content = ([{"type": "video", "video": str(root / job["video_file"])}] if video else [])
        content.append({"type": "text", "text": prompt})
        for attempt in range(3):
            inputs = tensors = generated = None
            stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                rendered = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
                assert ("<|video_pad|>" in rendered) == video
                if thinking:
                    assert rendered.rstrip().endswith("<think>")
                if video:
                    inputs = processor(text=[rendered], videos=[str(root / job["video_file"])], return_tensors="pt", cap_pixels_per_frame=True, return_metadata=True)
                    metadata = inputs.pop("video_metadata")
                    assert len(metadata) == 1
                    entry = metadata[0]
                    indices = entry.get("frames_indices") if isinstance(entry, dict) else entry.frames_indices
                    indices = [int(x) for x in indices]
                    assert indices
                    grid = inputs["video_grid_thw"].tolist()
                else:
                    inputs = processor(text=[rendered], return_tensors="pt")
                    assert all(inputs.get(k) is None for k in ["pixel_values", "pixel_values_videos", "video_grid_thw"])
                    indices = grid = None
                tensors = {k: v.to("cuda:0") for k, v in inputs.items()}
                generation = dict(max_new_tokens=limit, do_sample=False)
                if thinking:
                    generation.update(num_beams=1, use_cache=True)
                with torch.inference_mode():
                    generated = model.generate(**tensors, **generation)
                ids = generated[0, tensors["input_ids"].shape[-1]:].tolist()
                ended = bool(ids and ids[-1] in eos)
                limited = len(ids) >= limit and not ended
                if thinking:
                    raw = processor.tokenizer.decode(ids[:-1] if ended else ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                else:
                    raw = processor.batch_decode([ids], skip_special_tokens=True)[0]
                reasoning, final, status = split(raw, thinking, limited)
                torch.cuda.synchronize()
                row = {**job, "model_tag": tag, "model_version": config["model_version"], "prompt_text": prompt, "raw_answer": raw, "final_answer": final, "thinking": reasoning, "inference_status": status, "num_frames": len(indices) if indices else None, "sampled_frame_indices": indices, "video_grid_thw": grid, "latency_s": round(time.perf_counter() - started, 4), "run_time": stamp, "output_tokens": len(ids), "max_new_tokens": limit, "rendered_input_sha256": hashlib.sha256(rendered.encode()).hexdigest()}
                path = output / job["exp"] / tag / f"{job['prompt_id']}_{job['input_mode']}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
                print(f"[{i}/{len(jobs)}] {key(job)} {status} {row['latency_s']}s", flush=True)
                break
            except Exception as exc:
                with (root / f"{tag}_{'smoke' if args.smoke else 'full'}_errors.jsonl").open("a") as f:
                    f.write(json.dumps({**job, "attempt": attempt + 1, "run_time": stamp, "inference_status": "error", "raw_answer": repr(exc)}, ensure_ascii=False) + "\n")
                inputs = tensors = generated = None
                torch.cuda.empty_cache()
                if attempt == 2:
                    raise
                time.sleep(2)


if __name__ == "__main__":
    main()
