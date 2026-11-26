# ClueProbe

> **持续更新维护中 / Work in progress.** 当前为预印本配套的首批代码，完整复现流程、数据下载入口及部分实验模块将持续补充。各模块状态见 [integration status](docs/integration-status.md)。

Code integration for MAIN, TEXT, LADDER, INDUCE and three-stage evaluation. Scientific results, full annotations, predictions and figures: **To be released**.

## Installation

Python 3.10+ on Linux is recommended. Use separate environments for model families; historical pinned requirements are in `environments/`. They are evidence of individual runtimes, not one compatible global environment.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Entry points

```bash
python scripts/get_answers.py --runner qwen-main-text --help
python scripts/get_answers.py --runner qwen2b --help
python scripts/get_answers.py --runner qwen8b-native --help
python scripts/get_answers.py --runner videollama3 --help
python scripts/get_answers.py --runner closed-official --help
python scripts/get_answers.py --runner trained-evaluation --help
python scripts/get_answers.py --runner intern-supplemental --help
python scripts/eval.py --help
python scripts/calculate.py --help
```

These commands show actual source options. No GPU job or paid request starts from `--help`. Model weights, released input manifests, original videos and version-matched runtime receipts must be supplied separately. Do not use the synthetic examples for scientific experiments.

The public closed-model adapter defaults to offline validation and uses official provider origins. See [closed API setup](closed_api/README.md). Historical relay results are not relabelled as official-API results.

## Workflows

- MAIN/TEXT: `runners/main_text/run_qwen.py` is the verified 8350 Thinking-era core, with explicit model revision and data paths. It is not the later 8351 four-card-only 32B modification. See [integration status](docs/integration-status.md).
- LADDER/INDUCE: model-specific adapters in `runners/ladder_induce/`. Qwen2B uses its recorded 2 FPS path; Qwen8B uses native processor settings; VideoLLaMA3 verifies its original MAIN processing core. Read [runtime assets](docs/runtime-assets.md) before launching.
- Supplemental InternVL/InternVideo: [portable run options](docs/supplemental.md), preserving frame counts and generation limits; GPU validation pending.
- Judge: `evaluation/judge.py` and the exact frozen `v11_prompts.json`. Stages: behavior, content, induction. Dry-run is default; `--execute` invokes the configured provider. Calibration source selection is not automatically accepted as human gold.
- Trained-model evaluation: `training/evaluate.py`; supply the original base model or trained adapter and input manifest. The assistant-tail loss modules preserve the actual training objective; complete training deployment is still being integrated.
- Structural audits: `tools/validate_manifest.py`, `tools/audit_results.py`, `tools/join_references.py`. They do not measure semantic accuracy. `scripts/calculate.py --workflow coverage` audits coverage; `--workflow y2y3` reproduces the final supplemental statistics from its frozen scored package. See [statistics reproduction](docs/statistics.md).

## Dataset download and placement

See [dataset setup](datasets/README.md) for the project-relative layout and per-run placement rules. The benchmark annotation download URL is **To be released**; no internal server path is a public download link. Original source videos are obtained under their datasets’ access terms. Example configuration: [paths.example.json](configs/paths.example.json).

## Data contracts

LADDER/INDUCE inputs contain exactly `exp, sample_id, item, variant, question, video_file, prompt_id, input_mode`. References stay separate and join by `(exp, sample_id, variant)`; inference keys additionally include prompt and input mode. MAIN/TEXT use the four fields `sample_id, video_id, question_zh, video_relative_path` and their own canonical input checks. Do not mix the schemas or old 1720/updated 1721 manifests.

## Provenance and release boundary

Selected execution sources are tracked by original SHA-256 in `metadata/source_hashes.json`. Historical diagnostics, failed LLaVA versions and private deployment paths remain in internal archives. Imported source is not evidence that this integration has been GPU-reproduced. Read [integration status](docs/integration-status.md) for the exact limitations.

Project license and author/citation metadata require maintainer confirmation. No license grant is inferred for third-party model implementations; obtain weights and upstream runtimes under their original terms.
