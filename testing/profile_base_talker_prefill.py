#!/usr/bin/env python3
"""
Capture the first talker prefill from a real Base-model generate() call, then profile
exactly one Qwen3TTSTalkerForConditionalGeneration.forward under torch.profiler.

Chrome trace: chrome://tracing or https://ui.perfetto.dev — load the exported JSON.
"""
from __future__ import annotations

import argparse
import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor


class StopAfterPrefillCapture(Exception):
    pass


def _clone_model_kwargs(kwargs: dict) -> dict:
    out = {}
    for k, v in kwargs.items():
        if torch.is_tensor(v):
            out[k] = v.detach().clone()
        else:
            out[k] = v
    return out


def capture_talker_prefill_kwargs(
    model: Qwen3TTSForConditionalGeneration,
    input_ids: list[torch.Tensor],
    languages: list[str],
) -> dict:
    captured: dict = {}
    orig_forward = model.talker.forward

    def _no_validate(self, model_kwargs):
        return

    model.talker._validate_model_kwargs = types.MethodType(_no_validate, model.talker)

    def wrapped(*args, **kwargs):
        ie = kwargs.get("inputs_embeds")
        if ie is not None and ie.ndim == 3 and ie.shape[1] > 1:
            captured["kw"] = _clone_model_kwargs(kwargs)
            raise StopAfterPrefillCapture()
        return orig_forward(*args, **kwargs)

    model.talker.forward = wrapped
    try:
        model.generate(
            input_ids=input_ids,
            languages=languages,
            max_new_tokens=64,
        )
    except StopAfterPrefillCapture:
        pass
    finally:
        model.talker.forward = orig_forward
        del model.talker._validate_model_kwargs
    if "kw" not in captured:
        raise RuntimeError(
            "Did not capture talker prefill; first talker.forward had no multi-token inputs_embeds."
        )
    return captured["kw"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=os.environ.get(
            "QWEN3_TTS_MODEL",
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ),
        help="HF repo id or local model directory",
    )
    p.add_argument(
        "--out",
        default=os.path.join(ROOT, "testing", "profiler_talker_prefill.json"),
        help="Chrome trace JSON path",
    )
    p.add_argument(
        "--text",
        default="Hello, this is a short profiling utterance.",
        help="Synthesis text (tokenized like inference)",
    )
    p.add_argument("--warmup", type=int, default=1, help="Profiler warmup repeats (not traced)")
    p.add_argument("--repeat", type=int, default=1, help="Traced repeats inside profiler (default 1)")
    args = p.parse_args()

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=(
            "flash_attention_2" if device == "cuda" else "sdpa"
        ),
        device_map=device if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.config.talker_config.pad_token_id = model.config.talker_config.codec_pad_id
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model, fix_mistral_regex=True)

    formatted = (
        f"<|im_start|>assistant\n{args.text}<|redacted_im_end|>\n<|im_start|>assistant\n"
    )
    enc = processor.tokenizer(formatted, return_tensors="pt")
    input_ids = [enc["input_ids"].to(device)]

    with torch.inference_mode():
        prefill_kw = capture_talker_prefill_kwargs(model, input_ids, languages=["auto"])

    model.talker.rope_deltas = None

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(args.warmup):
            model.talker.rope_deltas = None
            model.talker(**prefill_kw)
            sync()

        model.talker.rope_deltas = None
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(args.repeat):
                model.talker.rope_deltas = None
                with torch.profiler.record_function("talker_prefill_one_forward"):
                    model.talker(**prefill_kw)
                sync()

        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        prof.export_chrome_trace(args.out)

    print(f"Chrome trace: {args.out}")
    sort_by = (
        "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    )
    print(prof.key_averages().table(sort_by=sort_by, row_limit=30))


if __name__ == "__main__":
    main()
