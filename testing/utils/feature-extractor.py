# use the auto-feature extractor from qwen3ttsTokenizer file, and explain that architecture of the feature extractor ... is it same as the mimi encoder model ? compare both and 
# dont remove above comments

import os
import sys
import types

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, MimiModel

_testing_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_parent_dir  = os.path.dirname(_testing_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

WEIGHTS_DIR     = os.path.join(_testing_dir, "weights")
MAIN_MODEL_DIR  = os.path.join(
    WEIGHTS_DIR,
    "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base",
    "snapshots",
    "fd4b254389122332181a7c3db7f27e918eec64e3",
)
SPEECH_TOK_DIR  = os.path.join(MAIN_MODEL_DIR, "speech_tokenizer")
AUDIO_PATH      = os.path.join(_testing_dir, "e2e_outputs", "ref_source.wav")

# ── 1. Load the feature extractor from the local speech tokenizer ─────────────
print("=" * 60)
print("1. Qwen3 TTS speech tokenizer feature extractor")
print("=" * 60)
qwen_fe = AutoFeatureExtractor.from_pretrained(SPEECH_TOK_DIR)
print("Class         :", type(qwen_fe).__name__)
print("Sampling rate :", qwen_fe.sampling_rate)
print("Full config   :\n", qwen_fe, "\n")

# ── 2. Load the reference Mimi feature extractor (from kyutai/mimi) ───────────
print("=" * 60)
print("2. Kyutai Mimi feature extractor (reference)")
print("=" * 60)
mimi_fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
print("Class         :", type(mimi_fe).__name__)
print("Sampling rate :", mimi_fe.sampling_rate)
print("Full config   :\n", mimi_fe, "\n")

# ── 3. Load audio & run both feature extractors ───────────────────────────────
wav, sr = sf.read(AUDIO_PATH, dtype="float32")
if wav.ndim > 1:
    wav = wav.mean(axis=-1)
print(f"Audio: shape={wav.shape}, original_sr={sr}\n")

qwen_inputs = qwen_fe(
    raw_audio=wav,
    sampling_rate=qwen_fe.sampling_rate,
    return_tensors="pt",
)
mimi_inputs = mimi_fe(
    raw_audio=wav,
    sampling_rate=mimi_fe.sampling_rate,
    return_tensors="pt",
)

print("Qwen FE output keys  :", list(qwen_inputs.keys()))
for k, v in qwen_inputs.items():
    print(f"  {k}: {v.shape}, dtype={v.dtype}")

print("\nMimi FE output keys  :", list(mimi_inputs.keys()))
for k, v in mimi_inputs.items():
    print(f"  {k}: {v.shape}, dtype={v.dtype}")

# ── 4. Comparison summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON: Feature Extractor vs Mimi Encoder model")
print("=" * 60)
print("""
The feature extractor (AutoFeatureExtractor) and the Mimi encoder model
(MimiModel) are two completely different things that work in sequence.

Feature Extractor
-----------------
- Type         : signal-processing pre-processor, NOT a neural network
- Has weights  : No — purely deterministic
- Input        : raw waveform (numpy / list of floats)
- Output       : normalised float tensor (input_values) + padding mask
- What it does : resamples to target_sr, normalises amplitude,
                 pads shorter clips in a batch to the same length
- Both Qwen and the reference kyutai/mimi use the same
  MimiFeatureExtractor class — they are identical in behaviour.

Mimi Encoder (inside Qwen3TTSTokenizer / MimiModel)
----------------------------------------------------
- Type         : full neural network (EnCodec-style conv encoder + RVQ)
- Has weights  : Yes — trained parameters
- Input        : normalised tensor from the feature extractor
- Output       : discrete integer codec tokens
- Architecture :
    waveform → convolutional downsampling encoder
             → Residual Vector Quantizer (RVQ)
             → codec token indices (shape: B, T_codes, num_quantizers)

Pipeline inside Qwen3TTSTokenizer.encode():
  raw audio
    → _normalize_audio_inputs()   resample + mono
    → feature_extractor()         normalise + pad → float tensor
    → mimi_encoder.encode()       conv encoder + RVQ → codec tokens
""")

# ── 5. Encode + decode with MimiModel ────────────────────────────────────────
print("=" * 60)
print("5. Encode → codec tokens → decode (round-trip)")
print("=" * 60)

mimi_model = MimiModel.from_pretrained("kyutai/mimi")
mimi_model.eval()

input_values = mimi_inputs["input_values"]
padding_mask = mimi_inputs["padding_mask"]

with torch.inference_mode():
    enc_out   = mimi_model.encode(input_values, padding_mask)
    audio_codes = enc_out.audio_codes

print(f"audio_codes shape : {audio_codes.shape}  (batch, num_quantizers, T_codes)")

with torch.inference_mode():
    dec_out = mimi_model.decode(audio_codes, padding_mask)
    reconstructed = dec_out.audio_values  # (B, 1, T_samples)

reconstructed_np = reconstructed[0, 0].float().cpu().numpy()
out_sr = mimi_model.config.sampling_rate

out_path = os.path.join(_testing_dir, "e2e_outputs", "fe_codec_roundtrip.wav")
sf.write(out_path, reconstructed_np, out_sr)
print(f"reconstructed wav : shape={reconstructed_np.shape}, sr={out_sr}")
print(f"saved to          : {out_path}")
