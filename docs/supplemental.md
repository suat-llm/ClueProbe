# InternVL / InternVideo supplemental INDUCE

The portable entry point is `python scripts/get_answers.py --runner intern-supplemental --help`.
Supply `--model internvl` or `--model internvideo`, `--model-path`, `--manifest`, `--media-root`, and a fresh `--output-root`. The canonical manifest must contain 6794 input-only jobs. Add `--smoke` for the recorded three-condition technical smoke (video/P0, video/P1, none/P1). Use a separate output directory for full inference. No automatic resume is implemented; an existing result file is refused.

The original 32/128 frame, 448 pixel, BF16, greedy 128-token generation settings are retained. InternVL uses one GPU and SDPA; InternVideo uses two 24GiB GPUs, automatic placement and the checkpoint runtime defaults. Install checkpoint-specific upstream dependencies separately. The repository does not package remote model code or weights.

This derivative makes only deployment paths/configuration portable and strengthens termination checking: actual EOS is required instead of accepting an arbitrary short sequence. Invalid output is saved then execution stops; no silent re-generation or cap change. This is intentionally not a byte-identical replay of historical failure handling. Recorded historical four InternVL truncations remain unchanged in the archived results. These portable adapters have not been GPU-executed during integration.
