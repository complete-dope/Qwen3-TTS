import sys
import os
import json
import types
import time
import urllib.request
import io
import logging as _stdlib_logging

import torch
import numpy as np
import soundfile as sf
import librosa

# ── bootstrap `testing` as a package ────────────────────────────────────────
_testing_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir  = os.path.dirname(_testing_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_pkg = types.ModuleType("testing")
_pkg.__path__   = [_testing_dir]
_pkg.__package__ = "testing"
sys.modules.setdefault("testing", _pkg)

# ── patch broken imports in model.py ────────────────────────────────────────
from transformers.integrations import use_kernel_forward_from_hub as _ukffh
import transformers.utils as _tu
_tu.use_kernel_forward_from_hub = _ukffh

from transformers.utils import logging as _tf_logging
_stdlib_logging.get_logger = _tf_logging.get_logger

# ── patch missing "default" rope type ───────────────────────────────────────
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE
if "default" not in _ROPE:
    def _default_rope(config, device=None, seq_len=None, layer_type=None):
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        base = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        if device is not None:
            inv_freq = inv_freq.to(device)
        return inv_freq, 1.0
    _ROPE["default"] = _default_rope

# ── imports ──────────────────────────────────────────────────────────────────
from transformers import Qwen2Tokenizer, AutoConfig, AutoModel, AutoFeatureExtractor

from testing.model_configuration import Qwen3TTSConfig
from testing.model import Qwen3TTSForConditionalGeneration
from testing.tokenizer_configuration import Qwen3TTSTokenizerV2Config
from testing.tokenizer import Qwen3TokenizerModel

# ── paths ────────────────────────────────────────────────────────────────────
WEIGHTS_DIR      = os.path.join(_testing_dir, "weights")
MAIN_MODEL_DIR   = os.path.join(
    WEIGHTS_DIR,
    "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base",
    "snapshots",
    "fd4b254389122332181a7c3db7f27e918eec64e3",
)
SPEECH_TOK_DIR   = os.path.join(MAIN_MODEL_DIR, "speech_tokenizer")
OUT_DIR          = os.path.join(_testing_dir, "e2e_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── reference audio (same clip as examples/test_model_12hz_base.py) ─────────
REF_AUDIO_URL  = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
REF_TEXT       = (w
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)
SYN_TEXT       = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."
LANGUAGE       = "Auto"


# ── helpers ──────────────────────────────────────────────────────────────────
def load_audio_from_url(url: str) -> tuple[np.ndarray, int]:
    print(f"  Downloading reference audio from {url} ...")
    with urllib.request.urlopen(url) as resp:
        audio_bytes = resp.read()
    with io.BytesIO(audio_bytes) as f:
        audio, sr = sf.read(f, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return audio.astype(np.float32), int(sr)


def build_assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def build_ref_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n"


def tokenize(tokenizer, text: str, device) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt", padding=True)["input_ids"].to(device)
    return ids.unsqueeze(0) if ids.dim() == 1 else ids


# ── step 1 : load custom TTS model ───────────────────────────────────────────
def load_speech_tokenizer_weights(model_dir: str) -> Qwen3TokenizerModel:
    AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
    AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TokenizerModel)
    m = AutoModel.from_pretrained(model_dir)
    if not isinstance(m, Qwen3TokenizerModel):
        raise TypeError(f"AutoModel returned {type(m)}, expected Qwen3TokenizerModel.")
    return m


def encode_wav_to_codes(
    speech_model: Qwen3TokenizerModel,
    feature_extractor,
    wav: np.ndarray,
    sr: int,
    device: torch.device,
    speech_compute_dtype: torch.dtype,
):
    target_sr = int(feature_extractor.sampling_rate)
    if int(sr) != target_sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=int(sr), target_sr=target_sr)
    inputs = feature_extractor(
        [wav.astype(np.float32)],
        sampling_rate=target_sr,
        return_tensors="pt",
    )
    input_values = inputs["input_values"].to(device=device, dtype=speech_compute_dtype).squeeze(1)
    padding_mask = inputs["padding_mask"].to(device=device, dtype=torch.long)
    speech_model.eval()
    with torch.inference_mode():
        return speech_model.encode(
            input_values,
            padding_mask,
            return_dict=True,
        )


def decode_codes_to_wav(
    speech_model: Qwen3TokenizerModel,
    audio_codes: torch.Tensor,
    device: torch.device,
) -> tuple[list[np.ndarray], int]:
    if audio_codes.dim() == 2:
        audio_codes = audio_codes.unsqueeze(0)
    audio_codes = audio_codes.to(device=device, dtype=torch.long)
    speech_model.eval()
    with torch.inference_mode():
        dec = speech_model.decode(audio_codes, return_dict=True)
    wavs = [w.to(torch.float32).detach().cpu().numpy() for w in dec.audio_values]
    return wavs, int(speech_model.get_output_sample_rate())


def build_main_model(model_dir: str) -> Qwen3TTSForConditionalGeneration:
    print("Building Qwen3TTSForConditionalGeneration via AutoModel.from_pretrained ...")
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    model = AutoModel.from_pretrained(model_dir)
    if not isinstance(model, Qwen3TTSForConditionalGeneration):
        raise TypeError(f"AutoModel returned {type(model)}, expected Qwen3TTSForConditionalGeneration.")
    print("  ✓ main model loaded")
    return model


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.bfloat16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    print(f"Using device={device}, dtype={dtype}")

    # ── 1. load our custom main model ────────────────────────────────────────
    print("\n=== Step 1: Load custom TTS model ===")
    model = build_main_model(MAIN_MODEL_DIR)
    model = model.to(device).to(dtype)
    model.eval()

    # ── 2. attach generation config ──────────────────────────────────────────
    print("\n=== Step 2: Load generation config ===")
    gen_cfg_path = os.path.join(MAIN_MODEL_DIR, "generation_config.json")
    with open(gen_cfg_path) as f:
        generate_config = json.load(f)
    model.load_generate_config(generate_config)
    print("  ✓ generation config loaded")

    # ── 3. load speech tokenizer (same pattern as load_weights.py) ─────────────
    print("\n=== Step 3: Load speech tokenizer ===")
    speech_tok = load_speech_tokenizer_weights(SPEECH_TOK_DIR)
    speech_compute_dtype = torch.float32
    speech_tok = speech_tok.to(device).to(speech_compute_dtype)
    speech_fe = AutoFeatureExtractor.from_pretrained(SPEECH_TOK_DIR)
    model.load_speech_tokenizer(speech_tok)
    print(f"  ✓ speech tokenizer loaded  (input_sr={speech_tok.get_input_sample_rate()}, "
          f"output_sr={speech_tok.get_output_sample_rate()})")

    # ── 4. load text tokenizer ────────────────────────────────────────────────
    print("\n=== Step 4: Load text tokenizer ===")
    text_tok = Qwen2Tokenizer.from_pretrained(MAIN_MODEL_DIR, fix_mistral_regex=True)
    print(f"  ✓ text tokenizer loaded  (vocab_size={text_tok.vocab_size})")

    # ── 5. load & encode reference audio ─────────────────────────────────────
    print("\n=== Step 5: Encode reference audio ===")
    ref_wav, ref_sr = load_audio_from_url(REF_AUDIO_URL)
    print(f"  ref audio shape={ref_wav.shape}, sr={ref_sr}")
    sf.write(os.path.join(OUT_DIR, "ref_source.wav"), ref_wav, ref_sr)

    enc_out = encode_wav_to_codes(
        speech_tok, speech_fe, ref_wav, ref_sr, device, speech_compute_dtype
    )
    ref_codes = enc_out.audio_codes
    print(f"  ref codes shape={ref_codes[0].shape}")

    with torch.inference_mode():
        ref_rt = speech_tok.decode(ref_codes[0].unsqueeze(0).to(device), return_dict=True)
    ref_rt_wav = ref_rt.audio_values[0].to(torch.float32).cpu().numpy()
    ref_rt_path = os.path.join(OUT_DIR, "sanity_ref_codec_roundtrip.wav")
    sf.write(ref_rt_path, ref_rt_wav, speech_tok.get_output_sample_rate())
    print(f"  sanity: codec round-trip ref -> {ref_rt_path} (listen / should resemble ref)")

    # extract speaker embedding (needs 24 kHz)
    SPEAKER_SR = model.speaker_encoder_sample_rate
    ref_wav_24k = librosa.resample(ref_wav, orig_sr=ref_sr, target_sr=SPEAKER_SR)
    with torch.inference_mode():
        spk_emb = model.extract_speaker_embedding(ref_wav_24k, sr=SPEAKER_SR)
    print(f"  speaker embedding shape={spk_emb.shape}")

    # ── 6. build voice_clone_prompt dict ─────────────────────────────────────
    voice_clone_prompt = dict(
        ref_code          = [ref_codes[0].to(device)],
        ref_spk_embedding = [spk_emb.to(device)],
        x_vector_only_mode = [False],
        icl_mode           = [True],
    )

    # ── 7. tokenize synthesis text + reference text ───────────────────────────
    print("\n=== Step 6: Tokenize texts ===")
    syn_input_ids = tokenize(text_tok, build_assistant_text(SYN_TEXT), device)
    ref_input_ids = tokenize(text_tok, build_ref_text(REF_TEXT),       device)
    print(f"  syn_input_ids shape={syn_input_ids.shape}")
    print(f"  ref_input_ids shape={ref_input_ids.shape}")

    # ── 8. run model.generate ─────────────────────────────────────────────────
    print("\n=== Step 7: Generate (this may take a while on CPU) ===")
    gen_kwargs = dict(
        max_new_tokens       = 512,  # keep short for CPU test
        do_sample            = True,
        top_k                = 50,
        top_p                = 1.0,
        temperature          = 0.9,
        repetition_penalty   = 1.05,
        subtalker_dosample   = True,
        subtalker_top_k      = 50,
        subtalker_top_p      = 1.0,
        subtalker_temperature = 0.9,
    )

    t0 = time.time()
    with torch.inference_mode():
        talker_codes_list, _ = model.generate(
            input_ids=[syn_input_ids],
            ref_ids=[ref_input_ids],
            voice_clone_prompt=voice_clone_prompt,
            languages=[LANGUAGE],
            non_streaming_mode=True,
            **gen_kwargs,
        )
    t1 = time.time()
    print(f"  ✓ generation done in {t1-t0:.1f}s")
    print(f"  generated codes shape={talker_codes_list[0].shape}")

    # ── 9. prepend reference codes and decode ────────────────────────────────
    print("\n=== Step 8: Decode to waveform ===")
    ref_code_tensor = ref_codes[0].to(device=device, dtype=torch.long)
    gen_code_tensor = talker_codes_list[0].to(device=device, dtype=torch.long)
    combined = torch.cat([ref_code_tensor, gen_code_tensor], dim=0)

    wavs, fs = decode_codes_to_wav(speech_tok, combined, device)

    ref_len = int(ref_code_tensor.shape[0])
    wav_full = wavs[0]
    up = speech_tok.get_decode_upsample_rate()
    cut = min(ref_len * up, int(wav_full.shape[0]))
    wav_out = wav_full[cut:]

    gen_only_path = os.path.join(OUT_DIR, "sanity_gen_codes_only.wav")
    w_gen, fs_g = decode_codes_to_wav(speech_tok, gen_code_tensor, device)
    sf.write(gen_only_path, w_gen[0], fs_g)
    print(f"  sanity: generated codes only -> {gen_only_path}")
    print(f"  decoded wav shape={wav_out.shape}, sample_rate={fs}")

    # ── 10. save output ──────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "e2e_voice_clone.wav")
    sf.write(out_path, wav_out, fs)
    print(f"\n✓ Output saved to: {out_path}")
    print(f"  Duration: {len(wav_out)/fs:.2f}s  |  Sample rate: {fs} Hz")

    print("\n=== Step 9: ASR check (openai/whisper-small) ===")
    _asr_path = os.path.join(_testing_dir, "e2e_outputs", "asr.py")
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("e2e_outputs_asr", _asr_path)
        _asr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_asr)
        log_path = _asr.transcribe_wavs_to_log(
            OUT_DIR,
            [
                ("clone (gen only)", "sanity_gen_codes_only.wav"),
                ("clone (trimmed output)", "e2e_voice_clone.wav"),
                ("codec round-trip ref", "sanity_ref_codec_roundtrip.wav"),
            ],
        )
        with open(log_path, encoding="utf-8") as lf:
            for line in lf:
                print(f"  {line.rstrip()}")
        print(f"  (saved {log_path})")
    except Exception as e:
        print(f"  ASR failed (install transformers + whisper deps): {e}")


if __name__ == "__main__":
    main()
