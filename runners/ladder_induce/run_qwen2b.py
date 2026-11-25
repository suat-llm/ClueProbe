"""v11 Qwen3-VL-2B runner; retains the MAIN 2 FPS/BF16 setup.

Run from the experiment directory, where datasets/ contains the videos.
Smoke: --smoke. Full: omit --smoke after reviewing smoke outputs.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TAG = "Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
PROMPTS = {
    "P0": "Answer the following question based only on the video.\nQuestion: {question}\nGive a brief and direct answer.",
    "P1": 'Answer the question using only evidence visible in the video.\nIf the evidence is sufficient, answer directly.\nIf the evidence supports only a plausible guess, state the guess with clear uncertainty.\nIf the video does not provide enough evidence, answer: "Cannot be determined from the video."\nQuestion: {question}',
}
FIELDS = {"exp", "sample_id", "item", "variant", "question", "video_file", "prompt_id", "input_mode"}


def key(row):
    return tuple(row[k] for k in ("exp", "sample_id", "variant", "prompt_id", "input_mode"))


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def validate(jobs, root):
    seen = set()
    for row in jobs:
        assert set(row) == FIELDS, "Inference manifest must contain only input/routing fields"
        assert key(row) not in seen, "Duplicate inference key"
        seen.add(key(row))
        assert row["question"] and row["prompt_id"] in PROMPTS
        mode = row["input_mode"]
        assert mode in {"video", "none", "swap", "black"}
        if mode == "none":
            assert row["video_file"] is None
        else:
            p = Path(row["video_file"])
            assert not p.is_absolute() and ".." not in p.parts
            assert (root / p).is_file(), f"Missing video: {p}"
        if row["exp"] == "INDUCE":
            assert (row["variant"] == "v6") == (mode == "swap")
            assert (row["variant"] == "v7") == (mode == "black")


def smoke_jobs(jobs):
    # Cover every experiment/variant/prompt/input mode, independently.
    selected = {}
    for row in jobs:
        selected.setdefault(tuple(row[k] for k in ("exp", "variant", "prompt_id", "input_mode")), row)
    result = list(selected.values())
    # Also exercise a second complete ladder at the opposite end of the input
    # (the retained export orders A before G), without leaking labels to input.
    last_ladder = {}
    for row in jobs:
        if row["exp"] == "LADDER":
            last_ladder[row["variant"]] = row
    present = {key(r) for r in result}
    result.extend(r for r in last_ladder.values() if key(r) not in present)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / "data/inference_jobs.jsonl"
    jobs = read_rows(manifest)
    validate(jobs, root)
    if args.smoke:
        jobs = smoke_jobs(jobs)
    if args.validate_only:
        print(f"Validated {len(jobs)} jobs")
        return

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    import torch
    from importlib.metadata import version
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.video_utils import VideoMetadata

    output = root / ("smoke" if args.smoke else "results")
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / f"{TAG}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_dir = root / ("smoke_configs" if args.smoke else "configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_tag": TAG, "model_version": f"Qwen/{TAG}@{REVISION[:7]}",
        "model_path": str(args.model_path), "revision": REVISION,
        "fps": 2.0, "dtype": "bfloat16", "do_sample": False,
        "max_new_tokens": 128, "prompts": PROMPTS,
        "processor_overrides": "none; same as MAIN", "text_mode": "none",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "versions": {p: version(p) for p in ["torch", "transformers", "qwen-vl-utils", "decord"]},
        "hardware": {"gpu": torch.cuda.get_device_name(0), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "server": os.uname().nodename},
        "timing": "CUDA synchronized preprocessing and generation; excludes model loading",
    }
    config_path = config_dir / f"run_config_{TAG}.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, "Configuration changed; use a new output directory"
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    done = set()
    for path in output.glob(f"*/{TAG}/*.jsonl"):
        for row in read_rows(path):
            assert row["inference_status"] in {"ok", "truncated"}
            assert key(row) not in done
            done.add(key(row))
    if all(key(j) in done for j in jobs):
        print("All requested jobs already complete")
        return
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, dtype=torch.bfloat16).eval().to("cuda:0")
    processor = AutoProcessor.from_pretrained(args.model_path)
    config_eos = model.generation_config.eos_token_id
    eos = set(config_eos if isinstance(config_eos, list) else [config_eos])
    for i, job in enumerate(jobs, 1):
        if key(job) in done:
            continue
        prompt = PROMPTS[job["prompt_id"]].format(question=job["question"])
        content = []
        if job["input_mode"] != "none":
            content.append({"type": "video", "video": str(root / job["video_file"]), "fps": 2.0})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        for attempt in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
            inputs = generated = None
            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                frames = None
                sampled = grid = None
                if job["input_mode"] == "none":
                    inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda:0")
                else:
                    images, videos, _ = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True, image_patch_size=processor.video_processor.patch_size)
                    assert len(videos) == 1
                    frames = len(videos[0][1]["frames_indices"])
                    sampled = [int(x) for x in videos[0][1]["frames_indices"]]
                    inputs = processor(text=[text], images=images, videos=[v for v, _ in videos], video_metadata=[VideoMetadata(**m) for _, m in videos], do_sample_frames=False, padding=True, return_tensors="pt").to("cuda:0")
                    grid = inputs["video_grid_thw"].tolist()
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False, max_new_tokens=128)
                ids = generated[0, inputs.input_ids.shape[1]:]
                answer = processor.batch_decode([ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                assert answer.strip(), "Empty model response"
                truncated = len(ids) >= 128 and int(ids[-1]) not in eos
                torch.cuda.synchronize()
                result = {**job, "model_tag": TAG, "model_version": config["model_version"], "prompt_text": prompt, "raw_answer": answer, "final_answer": answer, "thinking": None, "inference_status": "truncated" if truncated else "ok", "num_frames": frames, "latency_s": round(time.perf_counter()-started, 4), "run_time": stamp, "sampled_frame_indices": sampled, "video_grid_thw": grid, "output_tokens": len(ids)}
                path = output / job["exp"] / TAG / f"{job['prompt_id']}_{job['input_mode']}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
                print(f"[{i}/{len(jobs)}] {key(job)} {result['inference_status']} {result['latency_s']}s", flush=True)
                break
            except Exception as exc:
                error_path = root / ("smoke_errors.jsonl" if args.smoke else "errors.jsonl")
                with error_path.open("a") as f:
                    f.write(json.dumps({**job, "inference_status": "error", "raw_answer": str(exc), "attempt": attempt+1, "run_time": stamp}, ensure_ascii=False) + "\n")
                if inputs is not None: del inputs
                if generated is not None: del generated
                torch.cuda.empty_cache()
                if attempt == 2:
                    raise RuntimeError("Three attempts failed; stopped for inspection, successful rows retained") from exc
                time.sleep(2)


if __name__ == "__main__":
    main()
