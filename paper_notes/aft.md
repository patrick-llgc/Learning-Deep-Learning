# [An Attention Free Transformer](https://arxiv.org/abs/2105.14103) 

_September 2023_

tl;dr: A new mechanism to replace dot-product attention, by introducing a learned pair-wise position bias. No attention map!

#### Overall impression
Conventional scaled dot-product attention mechanism has quadratic time and space complexity wrt the context size. Many previous work (such as linear attention in [Transformers are RNNs](transformers_are_rnns.md)) try to approximate full attention operation.

In AFT, K and V (context) are first combined together with a set of **learned position biases**. This step generates a reduced context, akin to the compression of dictionary. The lookup of query in this dictionary is then performed by element wise multiplication.

AFT maintains direct interaction between any two points in the context, a major advantage of dot product attention. 

#### Key ideas
- AFT is a plugin replacement of MHA.
$$
Y_t = \sigma_q(Q_t) \odot \frac{\sum_{t'=1}^T \exp(K_{t'} + w_{t, t'}) \odot V_{t'}}{\sum_{t'=1}^T \exp(K_{t'} + w_{t, t'})}
$$
	- $\sigma_q$ is sigmoid. $\odot$ is elementwise product. $w \in R^{T \times T}$ is the learned position bias. 
	- w has no channels and is only positional dependet.
	- For each target position t, AFT performs a weighted average of values (element wise weighted), the result of which is combined with the query with elementwise mult. 
	- The weighting is simply keys and a set of learned positional biases.
- Attention map or attention matrix: rowwise softmax(QK^T), with TxT format.
	- Attention map gives the elementwise conenctivity and is computationally heavy O(T^2 d). It signifies for a given element in T seq, how much attention should it give to the weight in the weighted sum of V to generate the final result.
	- AFT eliminate the need of attention map.
- AFT-local: masked w as a [band matrix](https://en.wikipedia.org/wiki/Band_matrix), non-zero entries are confined to a diagonal band, comprising the main diagonal and zero or more diagonals on either side. When s=1, w is a tridiagnal matrix.
- AFT-simple: when s=0, no w is learned.
$$
Y_t = \sigma_q(Q_t) \odot \frac{\sum_{t'=1}^T \exp(K_{t'} ) \odot V_{t'}}{\sum_{t'=1}^T \exp(K_{t'} )} \\
= \sigma_q(Q_t) \odot \sum_{t'=1}^T（softmax(K) \odot V)_{t'}
$$
	- The context reduction is simpolied to elemetwise operation and global pooling.
	- There is no global connectivity
- AFT-conv: when $w_{t, t'}$ is only dependent on the relative position of t and t', f(t-t').
- When s=0, no positional bias. --> Global connectivity is lost. The paper does not state this clearly, but I think so.

#### Technical details
- The rearranged computational ordering of QKV is also found in [linearized attention](transformers_are_rnns.md) works. The difference is that AFT uses elementwise attention, while all linearized attention still use dot-product. This further reduces compuatational complexity from O(Td^2) to O(Td). --> This is not too much as typically T>>d.

#### Notes
- What keeps the global connectivity if the leanred positonal bias W=0?
- AFT can be viewed as performning attention where the number of attention heads is the same as the model's feature dimension. --> Why? I did not get it.
- [Post on Zhihu with very nice illustrations](https://zhuanlan.zhihu.com/p/656636957)