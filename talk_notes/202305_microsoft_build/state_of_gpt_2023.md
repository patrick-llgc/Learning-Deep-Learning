#[2023/05] State of GPT

+------------------------------------------------------------+
| By Andrej Karpathy                                         |
|                                                            |
| https://www.youtube.com/watch?v=bZQun8Y4L2A                |
|                                                            |
| How to train your ChatGPT assistant \-- An emerging recipe |
+------------------------------------------------------------+

# Opening/Ending Remarks by GPT
![](media/image001.jpg)

# Basic knowledge of GPT
![](media/image002.jpg)

![](media/image003.jpg)

![](media/image004.jpg)

![](media/image005.jpg)

![](media/image006.jpg)

LLaMA is a significantly more powerful model than GPT-3. LLaMA has much
fewer parameters, but are trained significantly longer. We should not
judge the power of a model just by the number of parameters.

![](media/image007.jpg)

![](media/image008.jpg)

![](media/image009.jpg)

This Shakespeare toy model seems to use letters as tokenization, as many
words are not in standard vocabulary in modern English.

![](media/image010.jpg)

Just in terms of predicting the next token, the model is forced to
understand a lot about the structure of the text and different concepts.

![](media/image011.jpg)

![](media/image012.jpg)

Base model vs Assistant model

GPT4 is a base model. ChatGPT backend is an assistant model.

The best base model available is LLaMA, but not commercially licensed.

Difference between base model and assistant model

Base model are not assistants. They just want to complete documents.

They have to be finetuned into \"assistants\" or promoted carefully.

One example is that, you can directly have a chat with ChatGPT, but you
need to trick (properly prompt) GPT base model to have a conversation
with you.

![](media/image013.jpg)

![](media/image014.jpg)

![](media/image015.jpg)

In SFT stage, nothing changed algorithmically. Swap out high quantity,
low quality data and use low quantity high quality data.

ChatGPT is an RLHF model.

Vicuna is an SFT model.

Base models, SFT model and RLHF models can all be deployed.

![](media/image016.jpg)

![](media/image017.jpg)

![](media/image018.jpg)

![](media/image019.jpg)

![](media/image020.jpg)

![](media/image021.jpg)

![](media/image022.jpg)

RLHF models are more polite but less creative.

![](media/image023.jpg)

# Applications

![](media/image024.jpg)

![](media/image025.jpg)

![](media/image026.jpg)

LLMs needs tokens to think. Only 80 layers so it cannot handle complex
reasoning. CoT gives LLM a chance to slow down the reasoning process.

![](media/image027.jpg)

LLM get stuck on a bad token and cannot recover. LLMs know when they
perform bad. So we have to make that up in the prompt to give it another
chance to sample and recover.

![](media/image028.jpg)

System 1 vs System 2

![](media/image029.jpg)

![](media/image030.jpg)

ToT is AlphaGo for text.

LLM does not want to succeed. They want to imitate.

![](media/image031.jpg)

![](media/image032.jpg)

![](media/image033.jpg)

Directly feed relevant information to prompt. Open book test for LLM.

![](media/image034.jpg)

![](media/image035.jpg)

SFT can be done right, but much more involved. RLHF is pretty much
research territory and not practical to do in industry.

![](media/image036.jpg)

![](media/image037.jpg)

![](media/image038.jpg)

# On Autonomous Driving

* How to tokenize a scene?

* How to sample different trajectories?

See [how to sample a language model](https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277). Rollout is a random process,
controlled by temperature.


* How to treat nav info?
