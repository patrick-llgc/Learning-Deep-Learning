# [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)

_August 2023_

tl;dr: Linear attention that allows for efficient training as a tranformer, AND efficient inference as an RNN.

#### Overall impression
RNN training needs backprop through time and has two issues. 1) Vanishing gradient. 2) Training cannot be parallelized. Yet RNN is efficient in inference, with linear scaling with time step (seq length).

Transformers can be trained in parallel but slow in inference. It scales quadratically with time step. 

> In industry, why RNN (conv-LSTM) beats Transformers for video processing? It is better to trade cloud computing power in training for edge computing power in inference. 

RWKV combines the efficient training of transformers and efficient inference of RNNs. It is not a RNN or transformer, but a CNN. It is a CNN over one dimensional seq of tokens. 

It stretches the notion of attention to the point that it is NOT really an attention.


#### Key ideas
- Summaries of the key ideas

#### Technical details
- Transformers can be paralleized on all (say, 50) tokens at the same time thanks to causal attention masks. For RNN, gitwe can only train one token at a time, as we cannot infer everything at the same time.
- Linear transformers vs [AFT](aft.md)

#### Notes
- [Youtube review by Yannic](https://www.youtube.com/watch?v=x8pW19wKfXQ)


#### Raw notes
- The entire prefix is fed into transformer to predict next token. This is called causal attention. A token can attend to all the tokens before it. This causes quadratic scaling of computation.
- Transformers vs RNNs
	- Transformers can only consider a limited number of tokens at a time. Essentially it forgets "completely forgets" beyond the context length.
	- Recurrent networks builds memory after looking at each token. RNN only requires constant memeory to do inference. However this is also the information bottleneck. We cannot explicitly consider any token that is way back, but rather rely on the hidden state bottleneck. RNN is also notoriously hard to train, coupled with vanishing gradient, and cannot be parallelized.
	- For a 50 token seq, transformers can be trained on the 50 token in parallel with the help from a causal mask. RNN can only be trained on the final token. Training efficiency is quite different.
- RWKV is a CNN across a one dimesnional seq of tokens. -- Yannic Kilcher
- Linear attention is a stretch, but RWKV is not the first to call it an attention mechanism. 魔改的太厉害，都不能算注意力机制了。Not even approximating the orignal mechanism.
- Attention mechanism Att(Q, K, V) = $Softmax(QK^T)V$.
- AFT: Replace interaction of Q,K with a fixed attention W (TxT-dim, T is seq len) across the dataset. This is too limiting, then there is a modulation k (key) calculated from input. For a word, the attention is the dataset-specific attention, plus a learned word-specific attention. **This is less powerful but more scalable than the original attention.**
- RWKV simplifies this further by consolidating the W into a vector w (channel wise, d-dim). RWKV assumes a fixed memory decay pattern of a linear drop-off, for all feature dimensions. RWKV in general forgets the past, but modulated by a sub-pattern of the current token.













