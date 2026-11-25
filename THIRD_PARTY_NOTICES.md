# Third-party dependencies

Model-specific experiments depend on Qwen3-VL, VideoLLaMA3, LLaVA-Video, InternVL and InternVideo checkpoints and upstream runtime code. Obtain these separately and follow each project's license and checkpoint terms. This repository does not relicense their weights or vendor implementations.

The original shared inference core under `runners/ladder_induce/adapters/` is included to preserve VideoLLaMA3 preprocessing identity; only the documented VideoLLaMA3 path is supported here. Its other historical model branches are not promoted to final experiment entry points.

Project license, contributor list and citation metadata remain subject to the maintainers' confirmation. No default MIT license is inferred.
