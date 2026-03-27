

# 3D RoPE : 
This is used in positional embedding 
https://www.youtube.com/watch?v=hCzJo4ui1P8

2d RoPE : 

Absolute positional embedding : so they make the core meaning of a word change and this is not what we want to happen so we introduced RoPE.. 




# Talker model 


# Speaker model 


# Tokenizer 
This includes encoder , decoder output model as datamodels subclassing from model output from torch.

Input to whole model : audio + reference_text 
Tokenizer input : Audio ?(maybe?)
Pretrained model as well is used in tokenizer which one and where are we using it ?

Causal means input at 'T' should depend on things that are '<= T' and nothing more

Conv transposed visualised : https://hannibunny.github.io/mlbook/neuralnetworks/convolutionDemos.html



Tokenizer : Just takes audio waveform as input and uses feature-extractor over top of it and get the values out

Streaming mode : low latency , text is arriving token by token , audio codec generates while text is arriving 

Non-streaming mode : typical API calls , where we have the full text before generation starts then we do a basic prepend for the text -> to generate all audio, higher latency as user waits for full audio generation 

# -----x------

Audio comes in microphone , the voltage value gets identified and that is done 24000 per second so we have an 24 khz as the freq

24 khz -> 24000 hz 

24000 samples per second of an audio

but we cant ask a model to predict these many samples 

So we compress these samples at an ratio of 1920, so we do this 24000/1920 -> 12.5 hz Tokenizer  

so now we are converting from samples to frames so we got down from 24000 samples to 12.5 frames for that 

1 frame = 1920 samples

So each frame (1920 samples grouped together) tell us something about that sound and we represent in a single integer called as code. 

Codebook size here we are using is of 3072 length so possible codes are out of 3072 

and one code compressing 1920 samples is not that good so we have 32 code books here 
that makes us output 32 codes in total so from 1920 samples we got to 32 codes and each code representing a different meaning that 32 code books are called RVQ ..  

# --x--- 

the core for generation remains same : LLM -> so create an embedding table --> then convert to 1024 dims -> pass to talker decoder model -> generates audio codes -> pass to RVQ (generate 31 more) [repeat] ->  tokenizer decoder -> waveform



