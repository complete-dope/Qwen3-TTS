# architecture of a 12 hz tokenizer model 
# testing this model by creating a small model by ourself

import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

import math
from dataclasses import dataclass 
from typing import Optional, Union, List, Tuple, Dict, Any , Callable
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers import MimiModel, MimiConfig

# wow xformers got so much new things!
from transformers.activations import ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.integrations import use_kernel_forward_from_hub
from transformers.utils import auto_docstring
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from transformers.utils import ModelOutput, auto_docstring, logging
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast



from testing.tokenizer_configuration import Qwen3TTSTokenizerV1EncoderConfig, Qwen3TTSTokenizerV1DecoderConfig


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Utils for doing GQA 
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# other type is called lazy attention (as this is done step by step and is not optimized for operations)
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key , module.num_key_value_groups)
    value_states = repeat_kv(value , module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


@dataclass
@auto_docstring
class Qwen3TTSTokenizerEncoderOutput(ModelOutput):
    r"""
    audio_codes (`List[torch.LongTensor]`):
        Discret code embeddings computed using `model.encode`, each tensor has shape (codes_length_i, num_quantizers).
    """

    audio_codes: List[torch.LongTensor] = None # encoder output is audio-codes
    # [12 ,123,421, 41 , ... ]


@dataclass
@auto_docstring
class Qwen3TTSTokenizerDecoderOutput(ModelOutput):
    r"""
    audio_values (`List[torch.FloatTensor]`):
        Decoded audio values, obtained using the decoder part of Qwen3TTSTokenizerV1.
        Each tensor has shape (segment_length_i).
    """

    audio_values: List[torch.FloatTensor] = None # decoder output is audio-value
    # [12.1 , 32.1 , 43.1 , 23.4 , ... ]

@auto_docstring
class Qwen3TTSTokenizerDecoderPreTrainedModel(PreTrainedModel):
    config: Qwen3TTSTokenizerV1DecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True


class Qwen3SpeechTokenizerCausalConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=None, stride=1, groups=1):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size, 
            stride = stride, 
            dilation = dilation,
            groups = groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size-1) * dilation + 1 # (3-1)*2 + 1 -> leaving out one gap of dilation  
        self.dilation = dilation  # not sure on how researchers figure out a nice value for this parameter 
        self.padding = padding
        

    def _get_extra_padding_for_conv1d(self, hidden_state: torch.Tensor) -> int:
        length = hidden_state.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal_length - length
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        extra_padding = self._get_extra_padding_for_conv1d(hidden_state)
        hidden_state = F.pad(hidden_state , (self.padding, extra_padding), mode="constant", value=0)
        return self.conv(hidden_state).contiguous()

# TDB later here we are doing causal / temporal conv 
# this is not causal BTW ... 
class Qwen3SpeechTokenizerCausalTransConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        # this is temporal convolution as we used 
        pad = kernel_size - stride # this is to ensure each pixel is preocesssed same no. of times (no partiality)
        self.left_pad = 0
        self.right_pad = int(pad)

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        if self.right_pad > 0:
            hidden_state = hidden_state[..., : hidden_state.shape[-1] - self.right_pad]
        return hidden_state.contiguous()

# residual connection + 1d conv block + depthwise conv (audio input)
class Qwen3SpeechTokenizerConvNeXtBlock(nn.Module):
    def __init__(self, dim:int) -> None:
        super().__init__()
        self.dwconv = Qwen3SpeechTokenizerCausalConvNet(
            dim,
            dim,
            kernel_size=7,
            groups=dim,
            dilation=1,
        ) # depthwise conv 
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim)) # parametersized so model can adjust this (how we have in layer norm ) 
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        input = hidden_state
        hidden_state = self.dwconv(hidden_state)
        hidden_state = hidden_state.permute(0, 2, 1)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.pwconv1(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.pwconv2(hidden_state)
        hidden_state = self.gamma * hidden_state
        hidden_state = hidden_state.permute(0, 2, 1)
        hidden_state = input + hidden_state
        return hidden_state


# TBD later
class Qwen3SpeechTokenizerDecoderRotaryEmbedding(nn.Module):
    def __init__(self, config , device=None) -> None:
        super().__init__()
        if hasattr(config , 'rope_scaling') and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        if self.rope_type == "default":
            inv_freq, self.attention_scaling = self.compute_default_rope_parameters(config, device)
        else:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def compute_default_rope_parameters(self, config=None, device=None):
        cfg = config if config is not None else self.config
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
        base = getattr(cfg, "rope_theta", 10000.0)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        if device is not None:
            inv_freq = inv_freq.to(device)
        return inv_freq, 1.0

    
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        
        return cos.to(dtype = x.dtype) , sin.to(dtype = x.dtype)

class Qwen3SpeechTokenizerDecoderAttention(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads # in reference to GQA ( grouped query attention paper) where no. of heads are splitted as per vanilla attn and we the KV heads are also limited / shared across the groups 
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = nn.Identity() # this is initialised to idenity cause we will be using LayerScale that learns a scaling parameter for each channel
        self.k_norm = nn.Identity()
        self.sliding_window = config.sliding_window

    def forward(
        self, 
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[torch.Tensor],
        cache_position: Optional[torch.Tensor], # whats this ? 
        **kwargs: Unpack[FlashAttentionKwargs],
    ):

        input_shape = hidden_states.shape[:-1] # B x T x D -> (B x T) 
        hidden_shape = (*input_shape, -1, self.head_dim) # (B ,T, -1, D)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2) # (B , -1, T, H)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2) # (B , -1, T, H)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (B , -1, T, H)

        cos,sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin) # TBD later

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)


        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen3SpeechTokenizerDecoderMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3SpeechTokenizerDecoderRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True) # square > mean > root
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

# LayerScale ? (https://huggingface.co/papers/2103.17239)
# deeper layer are much less effective than expected so we initialize a diagnol matrix that learns a channel scaling values for each channel  
# so the problem taht occurs in the deep stacks is while going till last layer we already have accumulate so much information that we dont get any more new meaningful information after that and last layers tend to learn nothing
class Qwen3SpeechTokenizerDecoderLayerScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.initial_scale = config.layer_scale_initial_scale
        self.scale = nn.Parameter(torch.full((self.hidden_size,), self.initial_scale, requires_grad=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x

# summation of whatall we did above :)  
class Qwen3SpeechTokenizerDecoderTransformerLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3SpeechTokenizerDecoderAttention(config, layer_idx)
        self.mlp = Qwen3SpeechTokenizerDecoderMLP(config)
        self.input_layernorm = Qwen3SpeechTokenizerDecoderRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3SpeechTokenizerDecoderRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn_layer_scale = Qwen3SpeechTokenizerDecoderLayerScale(config)
        self.mlp_layer_scale = Qwen3SpeechTokenizerDecoderLayerScale(config)
        self.attention_type = "sliding_attention"
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = residual + self.self_attn_layer_scale(hidden_states)
        
        # MLP 
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(hidden_states)
        return hidden_states


# this is one part of a decoder model , where we are using transformer 
class Qwen3SpeechTokenizerDecoderTransformerModel(Qwen3TTSTokenizerDecoderPreTrainedModel): # TBD: what is this tokenizer decoder 
    _can_record_outputs = {
        "hidden_states": Qwen3SpeechTokenizerDecoderTransformerLayer,
        "attentions": Qwen3SpeechTokenizerDecoderAttention,
    }

    def __init__(self, config: Qwen3TTSTokenizerV1DecoderConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen3SpeechTokenizerDecoderTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3SpeechTokenizerDecoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3SpeechTokenizerDecoderRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types # in 2026 we are stil using sliding attention ? 
        self.window_size = config.sliding_window
        
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)

        # Initialize weights and apply final processing
        self.post_init() #this initializes model from parent class but does it initialize random weights or some pretrained weights ? This boils down to `_init_weights` function from PreTrainedModel class and it initializes empty weights with mean and std 


    def forward(
        self,
        input_ids=None, 
        attention_mask=None, 
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,

    ) -> Qwen3TTSTokenizerDecoderOutput:
        if input_ids is not None: # input_ids can never pass through this filter
            raise ValueError("input_ids is not expected")
        if (input_ids is None) ^ (inputs_embeds is not None): # xor operator
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = self.input_proj(inputs_embeds)
        
        # TBD later ( cache scene )
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # positional embeddings 
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:# why is this check necessary? 
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
        



# Till now tbd are cache-position and rope and still I am not sure what going inside this model? is it text or its audio ? 
# ---------x---------

# now the activation for decoder 
class SnakeBeta(nn.Module):
    ''''
    This is used cause we want model to learn periodic functions , like audio waveforms ( this should be used for audio models as thats where we have this use case )

    this is a  type of activation function 
    Original snake function  : periodic function f(x) = x + 1/a * sin^2 (ax) , where a is a learnable parameter

    Modified snake function : uses alpha and beta 
    alpha for controlling frequency 
    beta for controlling magnitude
    
    '''
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features

        self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        self.no_div_by_zero = 0.000000001


    def forward(self, hidden_states):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        hidden_states = hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(hidden_states * alpha), 2)
        return hidden_states

# hm seems like this is an audio tokenizer that we are working with 
# rseidual connection in decoder only 
#what is this ? decoder residual unit ? what was decoder MLP then 
class Qwen3SpeechTokenizerDecoderResidualUnit(nn.Module):
    '''
    This model is trying to learn the wave periodic functions that we use in audio models to learn the waveform
    Here we are using conv net, so these hidden states should be of an audio model
    '''
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = Qwen3SpeechTokenizerCausalConvNet(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = Qwen3SpeechTokenizerCausalConvNet(dim, dim, kernel_size=1)

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.conv2(self.act2(self.conv1(self.act1(hidden_state))))
        return hidden_state + residual



# conv decoder block in decoder model  
# decoder - decoder block ? bad naming convention at least till now , its completely unreadable 
class Qwen3SpeechTokenizerDecoderDecoderBlock(Qwen3TTSTokenizerDecoderPreTrainedModel):
    def __init__(self, config: Qwen3TTSTokenizerV1DecoderConfig, layer_idx):
        '''
        architecture for this now looks like : 
        activation , conv net , act , conv , act , conv , act , conv .. like this we have multiple layers for this 
        '''
        super().__init__(config)
        in_dim = config.decoder_dim // 2**layer_idx
        out_dim = config.decoder_dim // 2 ** (layer_idx + 1)
        upsample_rate = config.upsample_rates[layer_idx]

        block = [
            SnakeBeta(in_dim),
            Qwen3SpeechTokenizerCausalTransConvNet(in_dim, out_dim, kernel_size=2 * upsample_rate, stride=upsample_rate)
        ]

        for dilation in (1, 3, 9):
            block.append(Qwen3SpeechTokenizerDecoderResidualUnit(out_dim, dilation))

        self.block = nn.ModuleList(block)

    def forward(self, hidden):
        for block in self.block:
            hidden = block(hidden)
        return hidden

'''
Representation learning : to produce some efficient representation of underlying data like VAE , that learns continuous latent space or we can have Discrete latent space also  

the discrete one is called vector quantization (VQ) mdoel where the codebooks are discrete  

* Vector Quantization : one codebook for vectors, closest prototype 

* Residual Vector Quantization : greedy decomposition,  multiple codebooks and finding the most relevant one residual connections

How do we do vector quantization in this ? 
https://www.youtube.com/watch?v=Xt9S74BHsvc
So this should be more like an encoder decoder model where in the bottleneck part we are doing the vector quantization and its more like EnCodec model 

* So residual vector quantization is just an recursive way of diving a space in smaller and smaller parts and this increases precision when constructing values back 

* So more the interations , more codebooks we need to learn  
so we get to this hyperparameter by deciding on our bitrate, so it best works like this, 12kbps bitrate results in 16 RVQ iterations and so on ...
'''

# In clustering also has now become an DL based problem
class EuclideanCodebook(nn.Module):
    '''
    Euclidean distance between 2 vectors 
    Codebook are also learned using k means clustering , that is we take common vectors together we create a mean value for it , that is referred to as codebook and then we select nearest vector in euclidean space for same 
    '''
    def __init__(self, dim: int, codebook_size: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.cluster_usage = nn.Parameter(torch.ones(codebook_size)) # initialized as all ones as all uniform   
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dim)) # initialized as all zeros  

    def decode(
        self,
        codes: torch.Tensor 
    ):
        embedding  = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        quantized = F.embedding(codes, embedding) # this just creates a look up for the indices that are in the codes and picks those up from the embedding matrix   
        return quantized


class VectorQuantization(nn.Module):
    
    def __init__(self, dim: int, codebook_size: int, codebook_dim: Optional[int] = None, epsilon: float = 1e-5):
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim

        requires_projection = codebook_dim != dim
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.epsilon = epsilon 
        self._codebook = EuclideanCodebook(dim=codebook_dim, codebook_size=codebook_size, epsilon=epsilon)
        self.codebook_size = codebook_size

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self._codebook.decode(codes)
        quantized = self.project_out(quantized)
        quantized = quantized.transpose(1, 2)
        return quantized

# multiple vector quantization layers with residual connection refers to this 
class ResidualVectorQuantization(nn.Module):
    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = torch.zeros([1], device=codes.device)[0]
        for idx, layer_codes in enumerate(codes):
            layer = self.layers[idx]
            assert isinstance(layer, VectorQuantization)
            quantized = quantized + layer.decode(layer_codes)
        return quantized


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self, 
        dimension: int = 128,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        n_q: int = 8,
        q_dropout: bool = False,
        no_quantization_rate: float = 0.0,
        bins: int = 1024,
        decay: float = 0.99,
        force_projection: bool = False,
    ):
        super().__init__()
        self.max_n_q = n_q
        self.n_q = n_q
        self.q_dropout = q_dropout
        self.no_quantization_rate = no_quantization_rate
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        self.bins = bins
        self.decay = decay
        self.input_proj: torch.nn.Module
        self.output_proj: torch.nn.Module
        if self.input_dimension == self.dimension and not force_projection:
            self.input_proj = torch.nn.Identity()
        else:
            self.input_proj = nn.Conv1d(self.input_dimension, self.dimension, 1, bias=False)

        if self.output_dimension == self.dimension and not force_projection:
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = nn.Conv1d(self.dimension, self.output_dimension, 1, bias=False)
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes) # vector quantization 
        quantized = self.output_proj(quantized) # then passing that through 1d conv to get final values out from this 
        return quantized
     

class SplitResidualVectorQuantizer(nn.Module):
    '''
    here we are splitting vector quantization into two parts, one for semantic and one for acoustic 
    so we are making 2 quantizer and both will serve different roles
    So we are making 2 codebooks and each of those have seperate meaning in this we are doing semantic and acoustic quantization  
    
    '''
    def __init__(
        self,
        *, 
        n_q : int = 8,
        n_q_semantic : int = 1,
        **kwargs,
    ):
        super().__init__()
        assert n_q > n_q_semantic, (
            f"Number of quantizers {n_q} must be larger "
            f"than the number of semantic quantizers {n_q_semantic}."
        )
        self.max_n_q = n_q
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic
        q_dropout = kwargs.pop("q_dropout", False)
        self.rvq_first = ResidualVectorQuantizer(
            n_q=n_q_semantic, force_projection=True, q_dropout=False, **kwargs
        )
        self.rvq_rest = ResidualVectorQuantizer(
            n_q=n_q - n_q_semantic,
            force_projection=True,
            q_dropout=q_dropout,
            **kwargs,
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized += self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return quantized

# Final decoder model Arrived ! 
class Qwen3TokenizerDecoder(Qwen3TTSTokenizerDecoderPreTrainedModel):
    def __init__(self, config: Qwen3TTSTokenizerV1DecoderConfig):
        super().__init__(config)
        self.total_upsample = np.prod(config.upsample_rates + config.upsampling_ratios) # product of all values in the array 
        self.pre_transformer = Qwen3SpeechTokenizerDecoderTransformerModel._from_config(config)

        self.quantizer = SplitResidualVectorQuantizer(
            dimension=config.codebook_dim // 2,
            n_q=config.num_quantizers,
            n_q_semantic=1,
            bins=config.codebook_size,
            input_dimension=config.codebook_dim,
            output_dimension=config.codebook_dim,
        )
        self.pre_conv = Qwen3SpeechTokenizerCausalConvNet(
            config.codebook_dim,
            config.latent_dim,
            kernel_size=3,
        )

        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3SpeechTokenizerCausalTransConvNet(config.latent_dim, config.latent_dim, factor, factor),
                        Qwen3SpeechTokenizerConvNeXtBlock(config.latent_dim),
                    ]
                )
            )

        self.upsample = nn.ModuleList(upsample)
        decoder = [Qwen3SpeechTokenizerCausalConvNet(config.latent_dim, config.decoder_dim, 7)]

        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3SpeechTokenizerDecoderDecoderBlock(config, i))
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder += [
            SnakeBeta(output_dim),
            Qwen3SpeechTokenizerCausalConvNet(output_dim, 1, 7),
        ]
        self.decoder = nn.ModuleList(decoder)
        self.post_init()

    def forward(self, codes):
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} layer of codes, got {codes.shape[1]}")
        
        hidden = self.quantizer.decode(codes)
        hidden = self.pre_conv(hidden).transpose(1, 2)
        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)

# this decoder was for audio codes so all that we did was for audio dataset (this needs to be analysed mapped out fully)
# in encoder we are doing nothing .. we are using all from the mimi model  
class Qwen3SpeechTokenizerEncoder(MimiModel):
    def __init__(
        self,
        config: MimiConfig
    ):
        super().__init__(config)
        self.config = config
        self.upsample = None
        self.decoder_transformer = None
        self.decoder = None
        self.post_init()

# mimi is a neural audio codec model with for speech representation and compression , operates at 1.1 kbps bitrate with a 12hz frame rate and has 16 RVQ ( more on this a bit later but we have an overall idea now)
# human has 39 bps


@auto_docstring
class Qwen3TTSTokenizerV2PreTrainedModel(PreTrainedModel):
    config: Qwen3TTSTokenizerV1DecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True


class Qwen3TokenizerModel(Qwen3TTSTokenizerV2PreTrainedModel):
    def __init__(
        self, 
        config: Qwen3TTSTokenizerV1DecoderConfig
    ):
        super().__init__(config)
        self.config = config
        self.encoder_valid_num_quantizers = config.encoder_valid_num_quantizers
        self.input_sample_rate = config.input_sample_rate
        self.output_sample_rate = config.output_sample_rate
        self.decode_upsample_rate = config.decode_upsample_rate
        self.encode_downsample_rate = config.encode_downsample_rate


        self.encoder = Qwen3SpeechTokenizerEncoder._from_config(config.encoder_config)
        
        self.decoder = Qwen3TokenizerDecoder._from_config(config.decoder_config)
        self.post_init()
        
    def get_model_type(self):
        return self.config.model_type
    
    def get_input_sample_rate(self):
        return self.input_sample_rate
    
    def get_output_sample_rate(self):
        return self.output_sample_rate
    
    def get_decode_upsample_rate(self):
        return self.decode_upsample_rate
    
    def get_encode_downsample_rate(self):
        return self.encode_downsample_rate
    
    def encode(
        self, 
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, Optional[torch.Tensor]], Qwen3TTSTokenizerEncoderOutput]:

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        encoded_frames = self.encoder.encode(
            input_values=input_values.unsqueeze(1),
            return_dict=True
        ) # this outputs the audio frames / codes  

        audio_codes = encoded_frames.audio_codes[:, :self.encoder_valid_num_quantizers]
        audio_codes = [code[..., :-(-mask.sum() // self.encode_downsample_rate)].transpose(0, 1) for code, mask in zip(audio_codes, padding_mask)]
        if not return_dict:
            return (audio_codes,)
        return Qwen3TTSTokenizerEncoderOutput(audio_codes)

    def decode(
        self, 
        audio_codes:torch.Tensor, 
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, Optional[torch.Tensor]], Qwen3TTSTokenizerDecoderOutput]:

        return_dict = return_dict if return_dict is not None else self.config.return_dict
    
        audio_lengths = (audio_codes[..., 0] > -1).sum(1) * self.decode_upsample_rate
    
        audio_codes = torch.clamp(audio_codes, min=0)   
        audio_values = self.decoder.chunked_decode(audio_codes.transpose(1, 2)).squeeze(1)

        audio_values = [a[:l] for a, l in zip(audio_values, audio_lengths)]

        if not return_dict:
            return (audio_values,)
        return Qwen3TTSTokenizerDecoderOutput(audio_values)

__all__ = ["Qwen3TokenizerModel", "Qwen3TTSTokenizerV2PreTrainedModel",
"Qwen3SpeechTokenizerEncoder", 
"Qwen3TokenizerDecoder"
]

