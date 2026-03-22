

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

