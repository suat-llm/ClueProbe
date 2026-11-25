# Dataset setup / 数据集放置说明

Benchmark annotation download: **To be released**. The public URL and release version will be added here after the team publishes the dataset. Do not use the empty legacy `questions.json` / `quick_questions.json` placeholders as experimental inputs.

项目内建议目录（相对仓库根目录）：

```text
datasets/
  annotations/                 # versioned benchmark input manifests and separate references
  videos/
    Charades_v1_480_mute/      # original required Charades MP4 files
    Ego4D_v2_540ss_mute/       # original required Ego4D MP4 files
models/                       # checkpoints obtained from their upstream providers
runs/experiment_name/         # writable model-specific runtime receipts and inputs
outputs/experiment_name/      # generated outputs; never committed
```

原始视频需按 Charades、Ego4D 等来源数据集的访问和许可条件自行获取。当前仓库尚未提供经团队确认的完整下载清单；发布后会补充版本、校验值和准确的获取入口。请勿将内部共享服务器路径当作公开下载地址。

MAIN/TEXT: pass `--input datasets/annotations/main_eval_1721.jsonl`, `--dataset-root datasets/videos`, and a separate `--output-root`. Each row's `video_relative_path` resolves under the dataset root. The canonical core has strict dataset checks; synthetic examples cannot replace the formal manifest.

InternVL/InternVideo supplemental: pass `--manifest`, `--media-root`, `--model-path`, `--output-root`; `video_file` resolves under the media root. See `docs/supplemental.md`.

Official closed API: `--run-root` contains `data/inference_jobs.jsonl` and the exact relative `video_file` layout in that manifest. Keep `--output-root` outside that run root. Validate before adding `--execute`.

Other LADDER/INDUCE adapters: a writable run root contains `data/inference_jobs.jsonl`, the relative video paths listed in it, and the identity receipts documented in `docs/runtime-assets.md`. Videos can be symlinked to project-local datasets to avoid copying. Do not edit manifest paths silently: record a new manifest hash if relocation changes them.

推理输入与参考答案分开保存。`data/examples/` 仅为合成格式示例。数据、原始作答、校准人工标签、权重及视频不包含在本次代码发布中。Results: **To be released**.
