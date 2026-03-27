# final modelling layer 
import torch 
import transformers
from ..model import Qwen3TTSForConditionalGeneration , Qwen3TTSTalkerModel 
from ..model_configuration import Qwen3TTSConfig

from ..tokenizer import Qwen3TokenizerModel , Qwen3TTSTokenizerV2PreTrainedModel ,Qwen3SpeechTokenizerEncoder , Qwen3TokenizerDecoder
from ..tokenizer_configuration import Qwen3TTSTokenizerV1EncoderConfig , Qwen3TTSTokenizerV1DecoderConfig

from transformers import MimiConfig
import librosa

#config
mimi_config = MimiConfig()

#dataset 
audio_file = '/Users/mohitdulani/Desktop/personal/audio-models/Qwen3-TTS/testing/dataset/ref_source.wav' 
audio_array, sr = librosa.load(audio_file, sr=mimi_config.sampling_rate)

print(type(audio_array))
print(len(audio_array))
tensor_array = torch.tensor(audio_array)

# tokenizer-encoder
tokenizer_encoder = Qwen3SpeechTokenizerEncoder(config=mimi_config)

encoded_frames = tokenizer_encoder.encode(tensor_array.unsqueeze(0).unsqueeze(1))
print('encoded frames shape : ' , encoded_frames.audio_codes.shape)
print(encoded_frames.audio_codes)
# from mimi model we are just using this encoder model .. 



# model 
Qwen3TTSForConditionalGeneration = Qwen3TTSForConditionalGeneration(config = Qwen3TTSConfig())
output = Qwen3TTSForConditionalGeneration(encoded_frames) # our trained one 

# tokenizer-decoder 
tokenizer_decoder = Qwen3TokenizerDecoder() # our own trained decoder 

