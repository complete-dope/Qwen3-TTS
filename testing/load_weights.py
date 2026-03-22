import sys
import os
import types
import glob
import logging as _stdlib_logging

import torch
from huggingface_hub import snapshot_download

# Bootstrap `testing` as a package so relative imports in model.py / tokenizer.py work
_testing_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_testing_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

_pkg = types.ModuleType("testing")
_pkg.__path__ = [_testing_dir]
_pkg.__package__ = "testing"
sys.modules.setdefault("testing", _pkg)

# model.py has two broken imports that need patching before the module loads:
# 1. `from transformers.utils import use_kernel_forward_from_hub`  (moved to integrations)
# 2. `logging.get_logger` used on stdlib logging instead of transformers logging
from transformers.integrations import use_kernel_forward_from_hub as _ukffh
import transformers.utils as _tu
_tu.use_kernel_forward_from_hub = _ukffh

from transformers.utils import logging as _tf_logging
_stdlib_logging.get_logger = _tf_logging.get_logger

# transformers 5.x dropped the "default" rope type — add it back
import torch as _torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE
if "default" not in _ROPE:
    def _default_rope(config, device=None, seq_len=None, layer_type=None):
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        base = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (
            base ** (_torch.arange(0, head_dim, 2, dtype=_torch.int64).float() / head_dim)
        )
        if device is not None:
            inv_freq = inv_freq.to(device)
        return inv_freq, 1.0
    _ROPE["default"] = _default_rope

from testing.model_configuration import Qwen3TTSConfig
from testing.tokenizer_configuration import Qwen3TTSTokenizerV2Config
from testing.model import Qwen3TTSForConditionalGeneration
from testing.tokenizer import Qwen3TokenizerModel

MAIN_MODEL_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
CACHE_DIR = os.path.join(_testing_dir, "weights")


def download_repo(repo_id: str) -> str:
    print(f"Downloading {repo_id} ...")
    local_dir = snapshot_download(repo_id, cache_dir=CACHE_DIR, token=HF_TOKEN)
    print(f"  -> {local_dir}")
    return local_dir


def load_state_dict_from_dir(model_dir: str) -> dict:
    from safetensors.torch import load_file

    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if shards:
        state = {}
        for s in shards:
            state.update(load_file(s, device="cpu"))
        return state

    shards = sorted(glob.glob(os.path.join(model_dir, "pytorch_model*.bin")))
    if shards:
        state = {}
        for s in shards:
            state.update(torch.load(s, map_location="cpu", weights_only=True))
        return state

    raise FileNotFoundError(f"No weight files (.safetensors or .bin) found in {model_dir}")


def load_main_model(model_dir: str) -> Qwen3TTSForConditionalGeneration:
    print("Building Qwen3TTSForConditionalGeneration from config ...")
    config = Qwen3TTSConfig.from_pretrained(model_dir)
    model = Qwen3TTSForConditionalGeneration(config)

    print("Loading state dict (strict=True) ...")
    state_dict = load_state_dict_from_dir(model_dir)
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys:
        print(f"  missing keys    : {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  unexpected keys : {result.unexpected_keys}")
    return model


def load_tokenizer_12hz(model_dir: str) -> Qwen3TokenizerModel:
    print("Building Qwen3TokenizerModel (12Hz) from config ...")
    config = Qwen3TTSTokenizerV2Config.from_pretrained(model_dir)
    model = Qwen3TokenizerModel(config)

    print("Loading state dict (strict=True) ...")
    state_dict = load_state_dict_from_dir(model_dir)
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys:
        print(f"  missing keys    : {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  unexpected keys : {result.unexpected_keys}")
    return model


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)

    main_model_dir = download_repo(MAIN_MODEL_REPO)
    tokenizer_dir = download_repo(TOKENIZER_REPO)

    print("\n--- 12Hz Tokenizer ---")
    tokenizer = load_tokenizer_12hz(tokenizer_dir)
    tokenizer.eval()
    print("Tokenizer loaded.\n")

    print("--- Main TTS Model ---")
    model = load_main_model(main_model_dir)
    model.eval()
    print("Main model loaded.\n")

    print("All weights loaded successfully with strict=True.")
