import sys, os, types, glob, logging as _log, threading, time, torch, soundfile as sf
from pathlib import Path
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import warnings
warnings.filterwarnings("ignore")

from safetensors.torch import load_file
from huggingface_hub import snapshot_download

TESTING = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTING)
sys.path.insert(0, os.path.dirname(TESTING))

_pkg = types.ModuleType("testing")
_pkg.__path__ = [TESTING]; _pkg.__package__ = "testing"
sys.modules.setdefault("testing", _pkg)

from transformers.integrations import use_kernel_forward_from_hub as _ukffh
import transformers.utils as _tu; _tu.use_kernel_forward_from_hub = _ukffh
from transformers.utils import logging as _tfl; _log.get_logger = _tfl.get_logger
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE
if "default" not in _ROPE:
    def _default_rope(config, device=None, **kw):
        hd = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv = 1.0 / (getattr(config, "rope_theta", 1e4) ** (torch.arange(0, hd, 2).float() / hd))
        return (inv.to(device) if device else inv), 1.0
    _ROPE["default"] = _default_rope

from testing.model_configuration import Qwen3TTSConfig
from testing.model import Qwen3TTSForConditionalGeneration
from testing.tokenizer import Qwen3TokenizerModel
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_cfg_v2", os.path.join(
    os.path.dirname(TESTING), "qwen_tts/core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py"))
_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
Qwen3TTSTokenizerV2Config = _mod.Qwen3TTSTokenizerV2Config
from transformers import Qwen2TokenizerFast

VOICE_DESIGN_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
_ROOT = Path(TESTING).resolve().parent
_DEFAULT_LOCAL = _ROOT / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
_HF_REPO_CACHE = Path.home() / ".cache/huggingface/hub" / f"models--{VOICE_DESIGN_REPO.replace('/', '--')}"


def _resolve_voice_design_dir() -> str:
    env = os.environ.get("QWEN_VOICE_DESIGN_DIR")
    if env and Path(env, "model.safetensors").is_file():
        return os.path.abspath(env)
    if (_DEFAULT_LOCAL / "model.safetensors").is_file():
        return str(_DEFAULT_LOCAL)
    return snapshot_download(VOICE_DESIGN_REPO)


MODEL_DIR = _resolve_voice_design_dir()
print("MODEL_DIR:", MODEL_DIR, flush=True)

_snap = Path(MODEL_DIR)
_ckpt = _snap / "model.safetensors"
_hf_blobs = _HF_REPO_CACHE / "blobs"
_incomplete = list(_hf_blobs.glob("*.incomplete")) if _hf_blobs.is_dir() else []
if not _ckpt.is_file():
    print(
        "\nNo model.safetensors here:\n"
        f"  {_ckpt}\n"
        f"  (optional HF cache issues: {len(_incomplete)} .incomplete blob(s) under hub cache)\n"
        "  Put weights in Qwen3-TTS-12Hz-1.7B-VoiceDesign/ next to this repo, set QWEN_VOICE_DESIGN_DIR, or run testing/download_voice_design.py\n",
        flush=True,
    )
    sys.exit(1)
if (
    _incomplete
    and _HF_REPO_CACHE.exists()
    and _snap.resolve().is_relative_to(_HF_REPO_CACHE.resolve())
):
    print(f"warning: {len(_incomplete)} .incomplete blob(s) in HF cache for this repo.\n", flush=True)

SPEECH_TOK_DIR = os.path.join(MODEL_DIR, "speech_tokenizer")
OUT_WAV = os.path.join(TESTING, "output_voice_design.wav")
# device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
device = 'cpu'

def _load(d):
    state = {}
    for s in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        state.update(load_file(s, device=device))
    return state

cfg = Qwen3TTSConfig.from_pretrained(MODEL_DIR)
if getattr(cfg, "tts_model_type", None) != "voice_design":
    print(f"warning: config tts_model_type={getattr(cfg, 'tts_model_type', None)!r} (expected 'voice_design')", flush=True)
cfg.talker_config.pad_token_id = cfg.talker_config.codec_pad_id

# MODEL : CONDITIONAL GENERATION MODEL 
model = Qwen3TTSForConditionalGeneration(cfg)
model.load_state_dict(_load(MODEL_DIR), strict=True)
model.eval()

# TOKENIZER : SPEECH TOKENIZER 
speech_tokenizer_config = Qwen3TTSTokenizerV2Config.from_pretrained(SPEECH_TOK_DIR)
speech_tok = Qwen3TokenizerModel(speech_tokenizer_config)
speech_tok.load_state_dict(_load(SPEECH_TOK_DIR), strict=True)
speech_tok.eval()

# TOKENIZER : TEXT TOKENIZER 
text_tok = Qwen2TokenizerFast.from_pretrained(MODEL_DIR)

TEXT = "ठीक है। हाँ। मुझे तुमसे नाराज़गी है। मैं तुमसे प्यार करता हूँ। मैं तुम्हारा सम्मान करता हूँ।"
INSTRUCT = (
    "Speak with a warm Mexican Spanish accent: natural Mexico Spanish, clear articulation, "
    "friendly conversational tone, slight local intonation, not overly formal."
)

formatted_text = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
formatted_instruct = f"<|im_start|>user\n{INSTRUCT}<|im_end|>\n"

input_ids = [text_tok(formatted_text, return_tensors="pt")["input_ids"]]
instruct_ids = [text_tok(formatted_instruct, return_tensors="pt")["input_ids"]]

_hb_stop = threading.Event()

def _heartbeat() -> None:
    t0 = time.time()
    while not _hb_stop.wait(20.0):
        print(f"  … generate still running ({time.time() - t0:.0f}s elapsed, 1.7B on {device} is slow)", flush=True)

_hb = threading.Thread(target=_heartbeat, daemon=True)
_hb.start()
try:
    with torch.inference_mode():
        code_list, _ = model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=["auto"],
        )
finally:
    _hb_stop.set()
    _hb.join(timeout=1.0)

print("Code list shape:", code_list[0].shape)

codes = code_list[0][:, :speech_tok.encoder_valid_num_quantizers].unsqueeze(0)
dec = speech_tok.decode(codes)
wav = dec.audio_values[0].detach().cpu().numpy()
sf.write(OUT_WAV, wav, speech_tok.output_sample_rate)
print(f"saved {OUT_WAV}  ({speech_tok.output_sample_rate} Hz, {len(wav)/speech_tok.output_sample_rate:.1f}s)")
