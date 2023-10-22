# [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)

_August 2023_

tl;dr: Linear attention (AFT) that allows for efficient parallelizable training as a tranformer, AND efficient inference as an RNN.

#### Overall impression
Transformers suffer from memory and computational complexity that scales quadratically with sequence length. The overarching motivation behind developing RWKV is to bridge the gap between computational efficiency and expressive capacity in neural network architecture. RWKV paves the way toward next-gen sustainable and computationally efficient AI models for seq processing tasks.

[RWKV](rwkv.md) stretches the notion of attention to the point that it is NOT really an attention but rather [AFT](aft.md). AFT can be seen as a MHA where each feature dimension corresponds to a head (n_channels == n_heads). Note that the **R in RWKV is essentially Q in AFT, and rebranded as receptance.**

[RWKV](rwkv.md) is very similar to [RetNet](retnet.md), achieving the impossible triangle of parallelizable training AND efficient inference AND Transformer-level language modeling quality.

Efficient, RNN-style inference means it's possible to run an int8 14B parameter RWKV model on sequences of any length with a constant memory requirement of 3GB VRAM. This opens up opportunities for language model-powered cognitive features in tightly-constrained edge environments with streaming inputs, like robotics, even if RWKV turns out, like other Transformer alternatives, to fall off the scaling laws eventually.


#### Key ideas
- RWKV leverages a linear attentio mechanism which allows formulation of the model as either a transformer or an RNN.
- RNN vs Transformers
	- RNN training needs backprop through time and has two issues. 1) Vanishing gradient. 2) Training cannot be parallelized. Yet RNN is efficient in inference, with linear scaling with time step (seq length).
	- Transformers can be trained in parallel but slow in inference. It scales quadratically with time step. 
	- In industry, it is better to trade cloud computing power in training for edge computing power in inference, so RNN (conv-LSTM) beats Transformers for video processing.
	- RWKV combines the efficient training of transformers and efficient inference of RNNs. It is not a RNN or transformer, but a CNN. It is a CNN over one dimensional seq of tokens. 
- Transformers vs AFT
	- Note that in the AFT Attn+, the denominator is a vector, and the division is an elementwise division.
$$
\text{Attn}(Q, K, V)_t = \frac{\sum_{i=1}^T \exp(q_t^T k_i) v_i}{\sum_{i=1}^T \exp(q_t^T k_i)} = \sum_{i=1}^T \frac{ \exp(q_t^T k_i)}{\sum_{i=1}^T \exp(q_t^T k_i)} v_i \\
= \sum_{i=1}^T \text{softmax}(q_t^T k_i)  v_i
$$ 

$$
\text{Attn}^+(W, K, V)_t = \frac{\sum_{i=1}^T \exp(w_{t, i} + k_i) \odot v_i}{\sum_{i=1}^T \exp(w_{t, i} + k_i)} = \sum_{i=1}^T \frac{\exp(w_{t, i} + k_i) }{\sum_{i=1}^T \exp(w_{t, i} + k_i)} \odot v_i \\
= \sum_{i=1}^T \text{softmax}(w_{t, i} + k_i) \odot v_i
$$

- R as receptance
	- Here WKV is rebranded as replacement of QKV in transformers, and the original Q is rebranded as R.
	- Sigmoid of receptance act as a "forget gate" to eliminate unnecessary historical info.
- W: From a learned positional bias matrix in [AFT](aft.md) to channel-wise time decay vector in [RWKV](rwkv.md).
	- In AFT-full, $w_{t,i} \in R^{T \times T}$. In AFT-conv, $w_{t,i} = f(t-i)$ and reduced w from TxT to 1xT. RWKV took it one step further to $w_{t,i} = -(t-i)w$, w has the shape of 1x1, and is non-negative. Then w is allowed to vary for diff channels. TxT --> 1xT --> 1x1 --> 1xd.
- Time mixing and channel mixing. See [A gist of RWKV MVP](https://gist.github.com/mattiasarro/c925e789e0358436f3e6c12731f5a196).
	- Forget about the token-shift, and time mixing is the MHA, and channel mixing is the FFN in transformer. 
	- The token-shift (time shift mixing) trick is not absolutely needed but the explicit form helps with time sequence modeling.
	- The token-shift is similar to the causal convolution in [WaveNet](https://arxiv.org/abs/1609.03499).

#### Technical details
- Transformers can be paralleized on all (say, 50) tokens at the same time thanks to causal attention masks. For RNN, gitwe can only train one token at a time, as we cannot infer everything at the same time.
- [Linear transformers](transformers_are_rnns.md) vs [AFT](aft.md)
- Why RWKV stands out from the rest of the efficient transformer papers?
	- Many alternative architectures have been proposed since the Transformer, from more efficient attention layers to reworked convolutional networks.
	- These alternatives generally show promising results up to a certain scale, say 1B parameters and 20B tokens, or >50x less than less than the current maximum scale for commercially available language models at time of writing (70B parameters, 2T tokens).
	- However, they have a reputation for falling off the scaling laws at some point shortly after.
	- The Receptance-Weighted Key-Value architecture, RWKV, has stayed on the scaling laws up to 14B parameters and 331B training tokens, which makes it, at time of writing, the largest-scale publicly-known non-Transformer generative language model. See the paper for details.

#### Notes
- [Youtube review by Yannic](https://www.youtube.com/watch?v=x8pW19wKfXQ)
- [AFT and RWKV with code on Bilibili](https://www.bilibili.com/video/BV1zW4y1D7Qg)
- [A gist of RWKV MVP](https://gist.github.com/mattiasarro/c925e789e0358436f3e6c12731f5a196)


#### Raw notes from Yannic's video
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









