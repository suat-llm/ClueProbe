# Official closed-model adapters

Official-interface adaptation for LADDER / INDUCE. Offline validation is the default. This is not the exact historical relay execution source and has not been validated through paid API calls.

## Setup

Python 3.10+, curl; OpenAI video-frame extraction additionally needs `pip install opencv-python`.
Credentials: `GEMINI_API_KEY` or `OPENAI_API_KEY` in the environment, or hidden interactive input. Never put credentials in commands or configuration files.

Select an exact model ID available in your official account. Historical relay names `gemini-3.8-flash` and `gpt-6.1-terra` are not assumed to be official IDs or equivalent models. Returned IDs must exactly match the requested ID; use a versioned ID where needed.

## Input and run

Prepare RUN_ROOT/data/inference_jobs.jsonl using exactly these fields:
`exp, sample_id, item, variant, question, video_file, prompt_id, input_mode`.
Video paths are relative to RUN_ROOT; none-mode uses null. References are forbidden in this input. Use a separate output directory outside RUN_ROOT.

```bash
python closed_api/run_official.py --provider google --model YOUR_OFFICIAL_MODEL_ID --run-root RUN_ROOT --output-root OUTPUT_ROOT --experiments INDUCE
```

This validates locally without credentials or network calls. Add `--execute --stage smoke` for a paid condition-covering smoke run. Add `--execute --stage pipeline` to run the same smoke gate followed by the full manifest. Reuse the same output directory to resume: input/config/model mismatches or invalid existing results stop execution.

Each run handles one provider/model. Google uses fixed generativelanguage.googleapis.com generateContent with original MP4 inlineData, maxOutputTokens=8192, temperature=0, default thinking. OpenAI uses fixed api.openai.com Chat Completions with 64 uniformly sampled original-resolution JPEG85 images, detail=auto, max_completion_tokens=2048, default temperature. No system messages or conversation history are added. Prompts are frozen in prompts.py.

Large inline videos may exceed provider request limits; failures stop rather than silently resize, change sampling, or upload through a different protocol. Requests are not automatically replayed after failure, because timeout/5xx may already have incurred charges. Concurrency is two. No built-in monetary cap: use provider project limits and a smoke run first. Optional VIDEOOOC_PRICING_JSON provides per-million token rates for estimates only.

Results retain usage, returned model, response, prompt, input/video SHA, timing and finish reason. Empty, blocked, truncated or model-mismatched responses are retained and stop the run. Existing historical relay results must use a different directory and are never imported into an official run.

## Historical scope

The team's final relay campaign used INDUCE video/swap/black, 3794 rows per model. LADDER and none were stopped; the earlier 6794/7784 plans must not be described as completed official runs. Gemini relay-channel injection and the accepted GPT alias belong to historical provenance. They are not applied as official-provider policies. This adapter can run LADDER if an explicit manifest is supplied, but no historical completion claim follows.

Results: **To be released**.

## Interface references

- https://platform.openai.com/docs/api-reference/chat/create
- https://ai.google.dev/gemini-api/docs/video-understanding

Verified against documentation on 2026-11-25. Exact model availability and official/relay equivalence remain unverified.
