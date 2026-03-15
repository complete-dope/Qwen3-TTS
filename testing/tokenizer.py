# architecture of a 12 hz tokenizer model 
# testing this model by creating a small model by ourself

import torch
import torch.nn as nn 
import torch.nn.functional as F


import math
from typing import Optional, Union, List, Tuple, Dict, Any , Callable
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.activations import ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel


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




class Qwen3SpeechTokenizerCausalConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation ,padding, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size, 
            stride = stride, 
            dilation = dilation, # default goes to 1 
        )
        self.stride = stride
        self.kernel_size = (kernel_size-1) * dilation + 1 # (3-1)*2 + 1 -> leaving out one gap of dilation  
        self.dilation = dilation  # not sure on how researchers figure out a nice value for this parameter 
        self.padding = padding
        

    def _get_extra_padding_for_conv1d(self, hidden_state: torch.Tensor) -> int:
        length = hidden_state.shape[-1] # B x C x L  (L)
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1 # classic conv formula (this can be a floating values as well) that tells the dimension value of the output , and here I am saying its frames 
        ideal_length = math.floor(n_frames) * self.stride + (self.kernel_size - self.padding) # not sure on this 
        return ideal_length - length
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        extra_padding = self._get_extra_padding_for_conv1d(hidden_state)
        hidden_state = F.pad(hidden_state , (self.padding, extra_padding), mode="constant", value=0)
        return self.conv(hidden_state).contiguous()

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
        hidden_state = self.dwconv(hidden_state) # B x C x L 
        hidden_state = hidden_state.permute(0, 2, 1) # B x L x C
        hidden_state = self.pwconv2(self.act(self.pwconv1(hidden_state)))
        hidden_state = self.gamma * hidden_state
        hidden_state = hidden_state.permute(0, 2, 1)
        hidden_state = input + hidden_state # residual connection
        return hidden_state


# TBD later
class Qwen3SpeechTokenizerDecoderRotaryEmbedding(nn.Module):
    def __init__(self, dim:int) -> None:
        
        pass

    def forward(self):
        pass

# still not sure fully what goes in the input ( need to be tokens of some sort [audio or text]) as we are doing attention  
class Qwen3SpeechTokenizerDecoderAttention(nn.Module):
    def __init__(self, config, layer_idx) -> None:
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
        cache_position: Optional[torch.Tensor],
        **kwargs: Unpack[FlashAttentionKwargs],
    ):

        input_shape = hidden_states.shape[:-1] # B x T x D -> (B x T) 
        hidden_shape = (*input_shape, -1, self.head_dim) # (B ,T, -1, D)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2) # (B , -1, T, H)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2) # (B , -1, T, H)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (B , -1, T, H)

        if past_key_values is not None:
            pass

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

class Qwen3SpeechTokenizerDecoderRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

# LayerScale ? ( )
# deeper layer are much less effective than expected so we initialize a diagnol matrix that learns this 


class Qwen3SpeechTokenizerStreaming(nn.Module):
    def __init__(self):
        super().__init__()

        pass 

class Qwen3SpeechTokenizerGeneratorAcousticCodebook(nn.Module):
    def __init__(self):
        ''''employs a 15-layer residual vector quantization (RVQ) module
        '''
        super().__init__()
        self.acoustic_model = ...  # 15-layer residual vector quantization (RVQ) module 

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.acoustic_codebook(waveform)

class Qwen3SpeechTokenizerGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_codebook = nn.Embedding(num_embeddings=1024, embedding_dim=1024)
        self.acoustic_codebook = nn.Embedding(num_embeddings=32, embedding_dim=1024)

        self.wavLM  = ... # model from microsoft and that acts as a teacher and is used to generate the semantic codebook values 

    def forward(self, text) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        text : (B, T_text)
        2 discrete code sequences : semantic codebook and acoustic codebook modelling acoustic details, prosody etc 
        '''

        return self.semantic_codebook(text), self.acoustic_codebook(text)

class Qwen3SpeechTokenizer(nn.Module):
    ''''   
    Gan based training framework , containing generator and discriminator  
    generator operates directly on raw waveforms to extract and quantize both representations, while the discriminator improves the naturalness and fidelity of reconstructed speech.
    '''
    def __init__(self):
        super().__init__()

        self.generator =  Qwen3SpeechTokenizerGenerator()# input waveform 
        self.discriminator = ... # TODO : what about this ? 


    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        self.generator(waveform)

        return 

reconstruction_loss = F.mse_loss(waveform, self.generator(waveform))

tokenizer_loss = reconstruction_loss 

# load the weights from huggingface model 


class Qwen3TextTokenizer(nn.Module):
    ''''
    Same as the text LLM one, used in qwen gpt models 
    '''
    pass


class SpeakerEncoder(nn.Module):
    ''''
    jointly trained ( with whom ?), a learnable speaker encoder with the backbone 
    '''
    def __init__(self):
        super().__init__()
        self.backbone = ... # TODO
        pass

