import sys, os, types, glob, logging as _log, torch, soundfile as sf
from safetensors.torch import load_file

TESTING = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTING)
sys.path.insert(0, os.path.dirname(TESTING))

# bootstrap testing as a package so internal imports in model.py / tokenizer.py work
_pkg = types.ModuleType("testing")
_pkg.__path__ = [TESTING]; _pkg.__package__ = "testing"
sys.modules.setdefault("testing", _pkg)

# patch transformers compat issues in model.py
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

MODEL_DIR = (
    "/Users/mohitdulani/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
    "/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3"
)
SPEECH_TOK_DIR = os.path.join(MODEL_DIR, "speech_tokenizer")

def _load(d):
    state = {}
    for s in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        state.update(load_file(s, device="cpu"))
    return state

# main model
cfg = Qwen3TTSConfig.from_pretrained(MODEL_DIR)
cfg.talker_config.pad_token_id = cfg.talker_config.codec_pad_id  # newer transformers requires this explicitly
model = Qwen3TTSForConditionalGeneration(cfg) # main model outputs code_list and hidden_states_list
model.load_state_dict(_load(MODEL_DIR), strict=True)
model.eval()

# speech tokenizer (decoder: codec codes → waveform)
speech_tokenizer_config = Qwen3TTSTokenizerV2Config.from_pretrained(SPEECH_TOK_DIR)
speech_tok = Qwen3TokenizerModel(speech_tokenizer_config)
speech_tok.load_state_dict(_load(SPEECH_TOK_DIR), strict=True)
speech_tok.eval()

# text tokenizer
text_tok = Qwen2TokenizerFast.from_pretrained(MODEL_DIR)
TEXT = "ठीक है। हाँ। मुझे तुमसे नाराज़गी है। मैं तुमसे प्यार करता हूँ। मैं तुम्हारा सम्मान करता हूँ।"
formatted = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
input_ids = [text_tok(formatted, return_tensors="pt")["input_ids"]]

# generate audio codes: code_list[0] shape (T, 32)
with torch.inference_mode():
    code_list, _ = model.generate(input_ids=input_ids, languages=["auto"])

print('Code list shape is : ', code_list[0].shape) # 

# decoder uses 16 quantizers only — take first 16 codes, shape (1, T, 16)
codes = code_list[0][:, :speech_tok.encoder_valid_num_quantizers].unsqueeze(0)
dec = speech_tok.decode(codes)
wav = dec.audio_values[0].detach().cpu().numpy()
sf.write("output.wav", wav, speech_tok.output_sample_rate)
print(f"saved output.wav  ({speech_tok.output_sample_rate} Hz, {len(wav)/speech_tok.output_sample_rate:.1f}s)")
