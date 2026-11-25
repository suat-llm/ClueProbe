"""v11 VideoLLaMA3 adapter preserving the corrected MAIN processing core."""
import argparse
import fcntl
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from run_qwen2b import PROMPTS, key, read_rows, validate, smoke_jobs

TAG = "VideoLLaMA3-7B"
REVISION = "d5b763e368861e7f5096e7ff1b49f92fbccf8ae6"
SHARED_SHA = "63dfbe664e6f37eeb2cf5eef1918b637a8e39a459198593e5eee44a62b6e3588"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    assert md.version("torch") == "2.5.1+cu121"
    assert md.version("transformers") == "4.46.3"
    assert shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg binaries must be in the task environment PATH"
    manifest = root / "data/inference_jobs.jsonl"
    jobs = read_rows(manifest)
    validate(jobs, root)
    if args.smoke:
        jobs = smoke_jobs(jobs)
    shared = root / "adapters/runner_a6000_snapshot.py"
    assert sha(shared) == SHARED_SHA, "Unexpected MAIN inference core"
    handoff = json.loads((root / "data/videollama3_main_handoff.json").read_text(encoding="utf-8"))
    for name, expected in handoff["checkpoint_metadata_sha256"].items():
        assert sha(args.model_path / name) == expected, f"Checkpoint metadata mismatch: {name}"
    spec = importlib.util.spec_from_file_location("main_snapshot", shared)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PROMPTS == PROMPTS
    import torch
    torch.cuda.set_device(0)
    torch.manual_seed(20261026)
    output = root / ("smoke" if args.smoke else "results")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / f"{TAG}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    done = set()
    expected = {key(j): j for j in jobs}
    for path in output.glob(f"*/{TAG}/*.jsonl"):
        for row in read_rows(path):
            assert key(row) not in done and key(row) in expected
            assert all(row[k] == expected[key(row)][k] for k in expected[key(row)])
            done.add(key(row))
    runner = module.make_runner(TAG, "cuda:0", {TAG: str(args.model_path)})
    started = time.perf_counter()
    runner.load()
    load_seconds = time.perf_counter() - started
    config = {
        "model_tag": TAG, "model_version": f"DAMO-NLP-SG/{TAG}@{REVISION}",
        "model_path": str(args.model_path),
        "revision_evidence": "current download metadata; historical full-weight equivalence not established",
        "generation": {"max_new_tokens": 128, "do_sample": False},
        "fps": 1, "max_frames": 64, "dtype": "bfloat16", "attention": "sdpa with MAIN per-frame segmentation patch",
        "processor": "native dynamic-resolution, add_generation_prompt=True",
        "prompts": PROMPTS, "seed": 20261026,
        "source_sha256": sha(Path(__file__)), "shared_runner_sha256": sha(shared),
        "manifest_sha256": sha(manifest),
        "checkpoint_metadata_sha256": handoff["checkpoint_metadata_sha256"],
        "versions": {p: md.version(p) for p in ["torch", "transformers", "torchvision", "tokenizers", "accelerate", "huggingface_hub", "safetensors", "numpy", "opencv-python-headless", "imageio"]},
        "host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe"),
        "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
    }
    cfg_dir = root / ("smoke_configs" if args.smoke else "configs")
    cfg_dir.mkdir(exist_ok=True)
    cfg = cfg_dir / f"run_config_{TAG}.json"
    if cfg.exists():
        assert json.loads(cfg.read_text(encoding="utf-8")) == config, "Configuration mismatch; do not mix runs"
    else:
        cfg.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Loaded {TAG} in {load_seconds:.2f}s", flush=True)

    stats = {}
    original_load = runner.processor.load_video
    def observe_load(*a, **kw):
        frames, timestamps = original_load(*a, **kw)
        stats.update(num_frames=len(frames), timestamps=[float(t) for t in timestamps])
        return frames, timestamps
    runner.processor.load_video = observe_load
    eos = runner.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for i, job in enumerate(jobs, 1):
        if key(job) in done:
            continue
        prompt = PROMPTS[job["prompt_id"]].format(question=job["question"])
        video = job["input_mode"] != "none"
        for attempt in range(3):
            stats.clear()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
            started = time.perf_counter()
            inputs = ids = None
            try:
                content = ([{"type": "video", "video": {"video_path": str(root / job["video_file"]), "fps": 1, "max_frames": 64}}] if video else [])
                content.append({"type": "text", "text": prompt})
                conv = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}]
                inputs = runner.processor(conversation=conv, return_tensors="pt", add_generation_prompt=True)
                tail = runner.processor.tokenizer.decode(inputs["input_ids"][0, -60:], skip_special_tokens=False)
                assert tail.endswith("<|im_start|>assistant\n"), repr(tail)
                shapes = {k: list(v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                grid = inputs["grid_sizes"].tolist() if video else None
                if video:
                    assert stats.get("num_frames", 0) > 0 and grid
                else:
                    assert not stats
                    assert all(inputs.get(k) is None or inputs[k].numel() == 0 for k in ["pixel_values", "grid_sizes"])
                inputs = {k: v.to(runner.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                with torch.inference_mode():
                    ids = runner.model.generate(**inputs, max_new_tokens=128, do_sample=False)
                answer = runner.processor.batch_decode(ids, skip_special_tokens=True)[0]
                token_ids = ids[0].tolist()
                truncated = len(token_ids) >= 128 and (not token_ids or token_ids[-1] not in eos)
                torch.cuda.synchronize()
                row = {**job, "model_tag": TAG, "model_version": config["model_version"], "prompt_text": prompt,
                       "raw_answer": answer, "final_answer": answer, "thinking": None,
                       "inference_status": "truncated" if truncated else "ok", "response_status": "valid" if answer.strip() else "invalid_empty",
                       "num_frames": stats.get("num_frames"), "sampled_timestamps_s": stats.get("timestamps"),
                       "video_grid_thw": grid, "input_shapes": shapes, "assistant_prefix_verified": True,
                       "latency_s": round(time.perf_counter() - started, 4), "run_time": stamp,
                       "output_tokens": len(token_ids), "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated()}
                path = output / job["exp"] / TAG / f"{job['prompt_id']}_{job['input_mode']}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
                print(f"[{i}/{len(jobs)}] {key(job)} {row['inference_status']} {row['latency_s']}s", flush=True)
                break
            except Exception as exc:
                with (root / f"{TAG}_{'smoke' if args.smoke else 'full'}_errors.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({**job, "attempt": attempt + 1, "error": repr(exc), "run_time": stamp}, ensure_ascii=False) + "\n")
                inputs = ids = None
                torch.cuda.empty_cache()
                if attempt == 2:
                    raise
                time.sleep(2)


if __name__ == "__main__":
    main()
