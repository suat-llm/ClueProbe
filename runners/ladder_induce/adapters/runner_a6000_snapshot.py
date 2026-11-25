#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video-OOC 预实验 P0/P1 推理脚本（逐样本计时）

用法：
  干跑（不加载模型，验证流程/计时/落盘）：
    python run_pilot.py --dry-run

  真实运行（在服务器上，可指定模型与 GPU）：
    python run_pilot.py --models Qwen3-VL-8B LLaVA-Video-7B --gpu 0
    python run_pilot.py --models all --gpu 0 --resume

输出（默认写在脚本同目录）：
  raw_predictions_quick60.jsonl  每条：sample_id/model/prompt_id/question/raw_answer/
                                 elapsed_seconds/start_ts/end_ts/inference_status
  timing_summary_quick60.csv     每个(模型,Prompt)的 N/total/mean/median/min/max 耗时

计时口径：elapsed_seconds 是单条“视频+问题+Prompt → 原始回答”的推理耗时
（不含模型加载时间；模型加载时间会单独打印）。
"""

import argparse
import csv
import datetime
import json
import os
import random
import statistics
import time


def _force_pyav_video_backend():
    """torchcodec 在本环境的 FFmpeg 下读不出视频流（“best video stream is unknown”）。
    这里强制 transformers 的视频解码改用 PyAV，绕开坏掉的 torchcodec。"""
    try:
        import transformers.video_processing_utils as vpu
        import transformers.video_utils as vu
    except Exception as e:
        print("[monkey-patch] 未应用（transformers 不可用）:", e)
        return

    _orig_load_video = vpu.load_video

    def _load_video_pyav(video_url_or_urls, backend=None, **kwargs):
        return _orig_load_video(video_url_or_urls, backend="pyav", **kwargs)

    vpu.load_video = _load_video_pyav
    print("[monkey-patch] 已强制 transformers 视频解码后端为 pyav")


_force_pyav_video_backend()


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(HERE, "dataset_en_quick_60.jsonl")

# P0/P1 提示词（严格对应手册 §5.1 / §5.2）
PROMPTS = {
    "P0": (
        "Answer the following question based only on the video.\n"
        "Question: {question}\n"
        "Give a brief and direct answer."
    ),
    "P1": (
        "Answer the question using only evidence visible in the video.\n"
        "If the evidence is sufficient, answer directly.\n"
        "If the evidence supports only a plausible guess, state the guess with clear uncertainty.\n"
        'If the video does not provide enough evidence, answer: "Cannot be determined from the video."\n'
        "Question: {question}"
    ),
}

# 手册 §4 的 6 个被测模型。hf_id 用于从 HuggingFace 加载。
MODELS = {
    "Qwen3-VL-2B":     {"hf": "Qwen/Qwen3-VL-2B-Instruct",   "kind": "transformers", "cap_pixels_per_frame": True, "model_class": "image_text_to_text"},
    "Qwen3-VL-8B":     {"hf": "Qwen/Qwen3-VL-8B-Instruct",   "kind": "transformers", "cap_pixels_per_frame": True, "model_class": "image_text_to_text"},
    "InternVL3.5-8B":  {"hf": "OpenGVLab/InternVL3_5-8B",    "kind": "official_repo"},
    "VideoLLaMA3-7B":  {"hf": "DAMO-NLP-SG/VideoLLaMA3-7B",  "kind": "videollama3"},
    "InternVideo2.5-8B": {"hf": "OpenGVLab/InternVideo2_5_Chat_8B", "kind": "official_repo"},
    "LLaVA-Video-7B":  {"hf": "lmms-lab/LLaVA-Video-7B-Qwen2", "kind": "llava_video"},
}


def load_dataset(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


class Runner:
    """统一接口：load() 加载一次；answer(video_path, prompt_text) 返回原始回答字符串。"""
    name = "base"

    def load(self):
        raise NotImplementedError

    def answer(self, video_path, prompt_text):
        raise NotImplementedError


class DummyRunner(Runner):
    name = "dummy"

    def load(self):
        pass

    def answer(self, video_path, prompt_text):
        time.sleep(random.uniform(0.02, 0.06))
        return "[dummy] " + prompt_text.split("Question:")[-1].strip()[:40]


def _resolve_model_class(model_class):
    if model_class == "llava_video":
        from transformers import LlavaVideoForConditionalGeneration
        return LlavaVideoForConditionalGeneration
    from transformers import AutoModelForImageTextToText
    return AutoModelForImageTextToText


def make_transformers_runner(name, hf_id, device, cap_pixels_per_frame=False, model_class="image_text_to_text"):
    """基于 transformers 的视频/图像多模态 VLM 加载器（Qwen3-VL / LLaVA-Video）。
    注意：不同模型的 processor 调用方式略有差异，请按各自模型卡调整。"""
    import importlib.util
    import os
    import torch
    from transformers import AutoProcessor

    class TransformersRunner(Runner):
        def load(self):
            processor_path = os.path.join(hf_id, "processing_videollama3.py")
            if os.path.isfile(processor_path):
                spec = importlib.util.spec_from_file_location(
                    "videollama3_local_processor", processor_path
                )
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                processor_cls = module.Videollama3Qwen2Processor
                original_get_args = processor_cls._get_arguments_from_pretrained.__func__

                @classmethod
                def compatible_get_args(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
                    return original_get_args(cls, pretrained_model_name_or_path, **kwargs)

                processor_cls._get_arguments_from_pretrained = compatible_get_args
                self.processor = processor_cls.from_pretrained(hf_id, trust_remote_code=True)
            else:
                self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
            model_cls = _resolve_model_class(model_class)
            self.model = model_cls.from_pretrained(
                hf_id, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
            )
            self.model.eval()

        def answer(self, video_path, prompt_text):
            # Qwen/LLaVA 系列需要套 chat template（视频 + 文本）
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt_text},
                ],
            }]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            proc_kwargs = dict(text=[text], videos=[video_path], return_tensors="pt")
            if cap_pixels_per_frame:
                proc_kwargs["cap_pixels_per_frame"] = True
            inputs = self.processor(**proc_kwargs).to(self.model.device)
            with torch.inference_mode():
                gen = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            gen = gen[:, inputs["input_ids"].shape[-1]:]
            return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    return TransformersRunner()


def make_official_repo_runner(model_name, hf_id):
    """需要官方仓库代码才能正确加载/处理视频的模型（InternVL3.5 / VideoLLaMA3 / InternVideo2.5）。
    这些模型的视频预处理不在标准 transformers 里，需按其官方 README 接入。"""
    hints = {
        "InternVL3.5-8B": "使用 OpenGVLab/InternVL 仓库的 internvl_chat（load_video + 自定义 chat template）。",
        "VideoLLaMA3-7B": "使用 DAMO-NLP-SG/VideoLLaMA3 仓库（AvHMTokenizer + 视频 token 处理）。",
        "InternVideo2.5-8B": "使用 OpenGVLab/InternVideo2.5 仓库的 demo 流程。",
    }

    class OfficialRepoRunner(Runner):
        def load(self):
            raise NotImplementedError(
                f"[{model_name}] 需按官方仓库接入视频处理器：{hints.get(model_name, '')} "
                f"HuggingFace: {hf_id}"
            )

        def answer(self, video_path, prompt_text):
            raise NotImplementedError

    return OfficialRepoRunner()


def make_videollama3_runner(name, hf_id, device):
    """VideoLLaMA3 专用 runner：AutoModelForCausalLM + processor(conversation=...)。"""
    import importlib.util
    import os
    import sys
    import types
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoProcessor

    # 兼容旧 VideoLLaMA3 代码：transformers 5.x 把 VideoInput 合并进了 ImageInput
    import transformers.image_utils as _image_utils
    if not hasattr(_image_utils, "VideoInput"):
        _image_utils.VideoInput = _image_utils.ImageInput

    # LLaVA-NeXT pins an older transformers API.  These helpers were moved or
    # removed in transformers 5.x; provide the unchanged legacy behavior
    # locally so the shared checkout remains untouched.
    import transformers.modeling_utils as _modeling_utils
    from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer
    if not hasattr(_modeling_utils, "apply_chunking_to_forward"):
        _modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    if not hasattr(_modeling_utils, "prune_linear_layer"):
        _modeling_utils.prune_linear_layer = prune_linear_layer
    if not hasattr(_modeling_utils, "find_pruneable_heads_and_indices"):
        def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            heads = set(heads) - already_pruned_heads
            mask = torch.ones(n_heads, head_size)
            for head in heads:
                head -= sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            index = torch.arange(mask.numel())[mask.view(-1).contiguous().eq(1)].long()
            return heads, index
        _modeling_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

    class VideoLLaMA3Runner(Runner):
        def load(self):
            processor_path = os.path.join(hf_id, "processing_videollama3.py")
            if os.path.isfile(processor_path):
                package_name = "videollama3_local"
                package = types.ModuleType(package_name)
                package.__path__ = [hf_id]
                sys.modules[package_name] = package
                module_name = f"{package_name}.processing_videollama3"
                spec = importlib.util.spec_from_file_location(module_name, processor_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                assert spec.loader is not None
                spec.loader.exec_module(module)
                processor_cls = module.Videollama3Qwen2Processor
                # Compat with transformers 5.x: the checkpoint's processor expects
                # a legacy common_kwargs bucket that is no longer declared upstream.
                from transformers.processing_utils import TextKwargs, ImagesKwargs, VideosKwargs, AudioKwargs
                processor_kwargs_cls = module.Videollama3Qwen2ProcessorKwargs
                processor_kwargs_cls.__annotations__.setdefault("text_kwargs", TextKwargs)
                processor_kwargs_cls.__annotations__.setdefault("images_kwargs", ImagesKwargs)
                processor_kwargs_cls.__annotations__.setdefault("videos_kwargs", VideosKwargs)
                processor_kwargs_cls.__annotations__.setdefault("audio_kwargs", AudioKwargs)
                processor_kwargs_cls.__annotations__.setdefault(
                    "common_kwargs", type("_VLLCommonKwargs", (), {"__annotations__": {}})
                )
                original_get_args = processor_cls._get_arguments_from_pretrained.__func__

                @classmethod
                def compatible_get_args(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
                    return original_get_args(cls, pretrained_model_name_or_path, **kwargs)

                processor_cls._get_arguments_from_pretrained = compatible_get_args
                self.processor = processor_cls.from_pretrained(hf_id, trust_remote_code=True)
            else:
                self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                trust_remote_code=True,
                device_map=device,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            self.model.eval()
            # The stock SDPA implementation constructs one quadratic attention mask
            # across every sampled frame.  The mask is block-diagonal, so evaluate
            # each independent frame segment directly to keep exact connectivity
            # without materializing the global matrix.
            def _segmented_sdpa_forward(attn, hidden_states, cu_seqlens, rotary_pos_emb=None):
                seq_length = hidden_states.shape[0]
                query_states = attn.q_proj(hidden_states).view(seq_length, attn.num_heads, attn.head_dim)
                key_states = attn.k_proj(hidden_states).view(seq_length, attn.num_heads, attn.head_dim)
                value_states = attn.v_proj(hidden_states).view(seq_length, attn.num_heads, attn.head_dim)
                query_states = module.apply_rotary_pos_emb_vision(query_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
                key_states = module.apply_rotary_pos_emb_vision(key_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
                boundaries = cu_seqlens.detach().cpu().tolist()
                outputs = []
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    q = query_states[start:end].transpose(0, 1)
                    k = key_states[start:end].transpose(0, 1)
                    v = value_states[start:end].transpose(0, 1)
                    outputs.append(F.scaled_dot_product_attention(q, k, v, dropout_p=0.0).transpose(0, 1))
                return attn.out_proj(torch.cat(outputs, dim=0).reshape(seq_length, -1))

            vision_layers = self.model.get_vision_encoder().encoder.layers
            vision_rotary = vision_layers[0].self_attn.__class__.forward.__globals__["apply_rotary_pos_emb_vision"]
            def _segmented_sdpa_forward(attn, hidden_states, cu_seqlens, rotary_pos_emb=None):
                seq_length = hidden_states.shape[0]
                query_states = attn.q_proj(hidden_states).view(seq_length, attn.num_heads, attn.head_dim)
                key_states = attn.k_proj(hidden_states).view(seq_length, attn.num_heads, attn.head_dim)
                value_states = attn.v_proj(hidden_states).view(seq_length, attn.num_heads, attn.head_dim)
                query_states = vision_rotary(query_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
                key_states = vision_rotary(key_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
                boundaries = cu_seqlens.detach().cpu().tolist()
                outputs = []
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    q = query_states[start:end].transpose(0, 1)
                    k = key_states[start:end].transpose(0, 1)
                    v = value_states[start:end].transpose(0, 1)
                    outputs.append(F.scaled_dot_product_attention(q, k, v, dropout_p=0.0).transpose(0, 1))
                return attn.out_proj(torch.cat(outputs, dim=0).reshape(seq_length, -1))
            for layer in vision_layers:
                if layer.self_attn.__class__.__name__ == "VisionSdpaAttention":
                    layer.self_attn.forward = types.MethodType(_segmented_sdpa_forward, layer.self_attn)

        def answer(self, video_path, prompt_text):
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 64}},
                    {"type": "text", "text": prompt_text},
                ]},
            ]
            inputs = self.processor(conversation=conversation, return_tensors="pt")
            inputs = {k: (v.to(self.model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return VideoLLaMA3Runner()


def make_llava_video_runner(name, hf_id, device, max_frames=64):
    """LLaVA-Video 专用 runner：用 LLaVA-NeXT 的 llava 代码库加载 + 推理。
    需在装有 decord 和 llava 包（pip install -e LLaVA-NeXT）的 legacy 环境里运行。"""
    import os
    import numpy as np
    import importlib.util
    import torch
    target_device = torch.device(device)
    torch.cuda.set_device(target_device)

    # 兼容补丁：LLaVA-NeXT(老 transformers) 从 modeling_utils import apply_chunking_to_forward，
    # 但新版把它挪到了 pytorch_utils，这里补一个别名。
    import transformers.modeling_utils as _modeling_utils
    from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer
    if not hasattr(_modeling_utils, "apply_chunking_to_forward"):
        _modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    if not hasattr(_modeling_utils, "prune_linear_layer"):
        _modeling_utils.prune_linear_layer = prune_linear_layer
    if not hasattr(_modeling_utils, "find_pruneable_heads_and_indices"):
        def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            heads = set(heads) - already_pruned_heads
            mask = torch.ones(n_heads, head_size)
            for head in heads:
                head -= sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            index = torch.arange(mask.numel())[mask.view(-1).contiguous().eq(1)].long()
            return heads, index
        _modeling_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM, LlavaQwenModel
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionConfig
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    from llava.utils import disable_torch_init
    from decord import VideoReader, cpu

    # The checkpoint predates transformers 5.x and stores rope_parameters as
    # null.  Preserve its recorded rope_theta while supplying the new schema.
    _original_config_from_pretrained = LlavaQwenConfig.from_pretrained

    @classmethod
    def _compatible_config_from_pretrained(cls, *args, **kwargs):
        cfg = _original_config_from_pretrained(*args, **kwargs)
        if getattr(cfg, "rope_parameters", None) is None:
            cfg.rope_parameters = {
                "rope_type": "default",
                "rope_theta": getattr(cfg, "rope_theta", 1_000_000.0),
            }
        return cfg

    LlavaQwenConfig.from_pretrained = _compatible_config_from_pretrained

    _original_model_from_pretrained = LlavaQwenForCausalLM.from_pretrained

    @classmethod
    def _compatible_model_from_pretrained(cls, *args, **kwargs):
        cfg = kwargs.get("config")
        if cfg is not None and getattr(cfg, "rope_parameters", None) is None:
            cfg.rope_parameters = {
                "rope_type": "default",
                "rope_theta": getattr(cfg, "rope_theta", 1_000_000.0),
            }
        return _original_model_from_pretrained(*args, **kwargs)

    LlavaQwenForCausalLM.from_pretrained = _compatible_model_from_pretrained

    _original_qwen_model_init = LlavaQwenModel.__init__

    def _compatible_qwen_model_init(self, config, *args, **kwargs):
        config.rope_parameters = {
            "rope_type": "default",
            "rope_theta": getattr(config, "rope_theta", 1_000_000.0),
        }
        return _original_qwen_model_init(self, config, *args, **kwargs)

    LlavaQwenModel.__init__ = _compatible_qwen_model_init

    if not hasattr(SigLipVisionConfig, "_set_token_in_kwargs"):
        SigLipVisionConfig._set_token_in_kwargs = staticmethod(lambda kwargs: None)

    FOR_GET_FRAMES_NUM = max_frames
    CONV_MODE = "qwen_1_5"

    class LlavaVideoRunner(Runner):
        def load(self):
            disable_torch_init()
            model_name = get_model_name_from_path(hf_id)
            overwrite_config = {
                "mm_spatial_pool_mode": "average",
                "mm_spatial_pool_stride": 4,
                "mm_newline_position": "no_token",
                "image_aspect_ratio": "pad",
            }
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                hf_id, None, model_name, load_8bit=False, device_map={"": target_device},
                torch_dtype="bfloat16", attn_implementation="sdpa", overwrite_config=overwrite_config
            )
            self.model.eval()
            self.add_time_instruction = getattr(self.model.config, "add_time_instruction", False)
            self.force_sample = getattr(self.model.config, "force_sample", False)
            self.mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)

        def _load_video(self, video_path):
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frame_num = len(vr)
            video_time = total_frame_num / vr.get_avg_fps()
            fps = round(vr.get_avg_fps())
            frame_idx = [i for i in range(0, len(vr), fps)]
            frame_time = [i / fps for i in frame_idx]
            if len(frame_idx) > FOR_GET_FRAMES_NUM or self.force_sample:
                uniform = np.linspace(0, total_frame_num - 1, FOR_GET_FRAMES_NUM, dtype=int)
                frame_idx = uniform.tolist()
                frame_time = [i / vr.get_avg_fps() for i in frame_idx]
            frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
            frames = vr.get_batch(frame_idx).asnumpy()
            return frames, frame_time, video_time

        def answer(self, video_path, prompt_text):
            frames, frame_time, video_time = self._load_video(video_path)
            video = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(device=target_device, dtype=torch.bfloat16)
            video = [video]

            qs = prompt_text
            if self.add_time_instruction:
                time_instruction = (
                    f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly "
                    f"sampled from it. These frames are located at {frame_time}. Please answer the following "
                    "questions related to this video."
                )
                qs = f"{time_instruction}\n{qs}"
            if self.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[CONV_MODE].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(target_device)
            if self.tokenizer.pad_token_id is None:
                if "qwen" in (self.tokenizer.name_or_path or "").lower():
                    self.tokenizer.pad_token_id = 151643
            attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().to(target_device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

            torch.cuda.empty_cache()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids, images=video, attention_mask=attention_masks,
                    modalities="video", do_sample=False, temperature=0.0,
                    max_new_tokens=256, top_p=0.1, num_beams=1, use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
            return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return LlavaVideoRunner()


def make_runner(model_name, device, overrides=None, llava_frames=64):
    overrides = overrides or {}
    if model_name == "dummy":
        return DummyRunner()
    cfg = MODELS.get(model_name)
    if cfg is None:
        raise KeyError(f"未知模型 {model_name}，可选: {list(MODELS)} + dummy")
    src = overrides.get(model_name) or cfg["hf"]
    if cfg["kind"] == "official_repo":
        r = make_official_repo_runner(model_name, src)
        r.name = model_name
        return r
    if cfg["kind"] == "videollama3":
        r = make_videollama3_runner(model_name, src, device)
        r.name = model_name
        return r
    if cfg["kind"] == "llava_video":
        r = make_llava_video_runner(model_name, src, device, max_frames=llava_frames)
        r.name = model_name
        return r
    r = make_transformers_runner(
        model_name, src, device,
        cap_pixels_per_frame=cfg.get("cap_pixels_per_frame", False),
        model_class=cfg.get("model_class", "image_text_to_text"),
    )
    r.name = model_name
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--models", nargs="+", default=["dummy"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--limit", type=int, default=None,
                    help="只跑前 N 条样本（用于单条/少量自测）")
    ap.add_argument("--start-index", type=int, default=0,
                    help="从数据集的零基位置开始；与 --limit、--resume 结合以小批次续跑")
    ap.add_argument("--model-path", action="append", default=[],
                    help="覆盖模型本地路径，格式 模型名=/本地/路径，可多次传入")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--llava-frames", type=int, default=64,
                    help="LLaVA-Video uniformly sampled frame cap; benchmark configuration is 64")
    args = ap.parse_args()

    if args.dry_run:
        args.models = ["dummy"]

    if args.models == ["all"]:
        args.models = list(MODELS.keys())

    overrides = {}
    for item in args.model_path:
        if "=" in item:
            name, path = item.split("=", 1)
            overrides[name.strip()] = path.strip()

    device = f"cuda:{args.gpu}" if not args.dry_run else "cpu"
    data = load_dataset(args.dataset)
    if args.start_index < 0:
        ap.error("--start-index must be non-negative")
    if args.limit is not None:
        data = data[args.start_index : args.start_index + args.limit]
    else:
        data = data[args.start_index :]
    print(f"数据集: {args.dataset}  样本数: {len(data)}")
    print(f"模型: {args.models}   Prompt: {list(PROMPTS)}   device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    run_config = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "purpose": "small LLaVA-Video pilot; not a replacement for the main benchmark run",
        "dataset": os.path.abspath(args.dataset),
        "start_index": args.start_index,
        "sample_count": len(data),
        "resume": args.resume,
        "models": args.models,
        "model_path_overrides": overrides,
        "prompts": PROMPTS,
        "device": device,
        "generation": {"do_sample": False, "max_new_tokens": 256, "num_beams": 1},
        "processor": {
            "name": "LLaVA-NeXT VideoReader and image processor",
            "llava_uniform_frame_cap": args.llava_frames,
        },
        "timing": {
            "field": "elapsed_seconds",
            "definition": "wall-clock time from just before video loading and prompt inference to decoded raw answer; excludes one-time model loading",
        },
    }
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "run_config_history.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(run_config, ensure_ascii=False) + "\n")

    from collections import defaultdict

    for model_name in args.models:
        raw_path = os.path.join(args.out_dir, f"raw_predictions_{model_name}.jsonl")
        timing_path = os.path.join(args.out_dir, f"timing_summary_{model_name}.csv")

        # 断点续跑：跳过已完成的 (model, prompt, sample)
        done = set()
        timing_rows = []
        if args.resume and os.path.exists(raw_path):
            with open(raw_path, encoding="utf-8") as f:
                for ln in f:
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    done.add((r["model"], r["prompt_id"], r["sample_id"]))
                    timing_rows.append(r)
            print(f"续跑：{model_name} 已跳过 {len(done)} 条已存在结果")

        raw_f = open(raw_path, "a" if args.resume else "w", encoding="utf-8")

        try:
            t_load0 = time.perf_counter()
            runner = make_runner(model_name, device, overrides, llava_frames=args.llava_frames)
            runner.load()
            load_s = time.perf_counter() - t_load0
            print(f"\n[模型] {model_name}  加载耗时 {load_s:.1f}s")

            for prompt_id, tmpl in PROMPTS.items():
                for row in data:
                    key = (model_name, prompt_id, row["sample_id"])
                    if key in done:
                        continue
                    prompt_text = tmpl.format(question=row["question_en"])
                    start_ts = datetime.datetime.now().isoformat(timespec="milliseconds")
                    t0 = time.perf_counter()
                    status = "ok"
                    try:
                        raw_answer = runner.answer(row["video_path"], prompt_text)
                    except Exception as e:  # 单条失败不中断整批
                        raw_answer = None
                        status = f"error: {type(e).__name__}: {e}"
                    elapsed = time.perf_counter() - t0
                    end_ts = datetime.datetime.now().isoformat(timespec="milliseconds")

                    rec = {
                        "sample_id": row["sample_id"],
                        "model": model_name,
                        "prompt_id": prompt_id,
                        "question": row["question_en"],
                        "raw_answer": raw_answer,
                        "elapsed_seconds": round(elapsed, 4),
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "inference_status": status,
                    }
                    raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    raw_f.flush()
                    timing_rows.append(rec)
                    print(f"  {model_name} {prompt_id} {row['sample_id']:<4} "
                          f"{elapsed:.3f}s  {status}")
        finally:
            raw_f.close()

        # 每个模型单独一份计时汇总
        agg = defaultdict(list)
        for r in timing_rows:
            if r["inference_status"] == "ok":
                agg[(r["model"], r["prompt_id"])].append(r["elapsed_seconds"])
        with open(timing_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "prompt_id", "n", "total_s", "mean_s", "median_s", "min_s", "max_s"])
            for (m, p), vals in sorted(agg.items()):
                w.writerow([
                    m, p, len(vals),
                    round(sum(vals), 3), round(statistics.mean(vals), 4),
                    round(statistics.median(vals), 4), round(min(vals), 4), round(max(vals), 4),
                ])
        print(f"\n[{model_name}] 完成。原始回答: {raw_path}")
        print(f"[{model_name}] 计时汇总: {timing_path}")


if __name__ == "__main__":
    main()
