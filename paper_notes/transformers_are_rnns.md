# [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)

_September 2023_

tl;dr: Kernel trick to approximate softmax attention allows matrix matmul with associative property and reduces quadratic attention complexity to linear, wrt to length.

#### Overall impression
Pain points of transformers: large memory, heavy computation. Actually the computation is not that heavy, and the computation is slow due to memory bandwith. The large memory increases the HW barrier for serving, and the heavy memory access pattern slows down the token generation.

Sparse attention or locality sensitivity hashing (LSH, Reformer) can reduce complexity from O(N^2) to O(N*sqrt(N)) or O(NlogN), but they do not speed up autoregressive inference. 

When we talk about new transformer architecture or mechanism, we need to see how it improves training or inference performance. For fast transformers, it is **linear complexity in computation and memory in training. Linear complexity in computation and constant memory in inference.**

The order of matmul matters a lot! The softmax hinders the application of associative property of matrix. Linear transformer removes the necessity of softmax and replaces that with element-wise activation (elu + 1). --> **This is the biggest contribution of this paper.** 

The self-attention is expressed as a linear dot-product of kernel feature maps. Then associativity property of matrix products is used to reduce the complexity to linear scale. This leads to 4000x speed up on autoregressive prediciton of iamges. 

#### Key ideas
- Softmax attention is only one kind of attention
	- $A_l(x) = V' = softmax(\frac{QK^T}{\sqrt D}) V$
	- Softmax is applied rowwise to QK^T.
	- Softmax attention is only one specific type of attention.
	- $sim(q, k) = \exp(q^T k / \sqrt D)$
	- As long as sim is non-negative, it can define an attention: $sim(x, y): R^{2 \times F} \rightarrow R_+$ 
- Kernel based formulation of self-attention
	- sim, the kernel can be the dot product of two feature representation.
	- $sim (Q_i, K_j) = \phi (Q_i) \phi (K_j)^T$, and the similarity function of the two vector generates a  scalar, which is used to weigh the corresponding row vector $V_j$. Note that the main text is wrong in this regarding the position of the T transposition. See Notes session below.
	- $\phi(x) = elu(x) + 1$, exponential linear unit activation function.
- Associative property of Matrix
	- The order of multiplication matters.
	- The dimension of the three matrix in Equation 6 is (NxD) (DxN) (NxM). 
	- If we do this sequantially, ((NxD) (DxN)) (NxM), then the computation is NxDxN + NxNxM = O(N^2 max(D, M)).
	- If we do the latter two first, (NxD) ((DxN) (NxM)), then the computation is O(DNM). When N >> D, then this method reduces complexity dramatically.
- Causal masking --> Isn't this very simlilar to KV cache?
	- if $S_i = \sum_{j=1}^i \phi(K_j) V_j$
	- if $Z_i = \sum_{j=1}^i \phi(K_j)$
	- then $V_i = \frac {\phi(Q_i) S_i}{\phi(Q_i) Z_i}$
	- $S_i$ and $Z_i$ can be computated autoregressively from $S_{i-1}$ and $Z_i-1$.
	- The inference can be computed with constant memory space and linear time.
	- The $S_i$ matrix $\phi(K_j) V_j$ can be considered the hidden state and updated at every time step like an RNN.

#### Technical details
- Summary of technical details

#### Notes
- Note that equation 4 and 5 has a typo with the position of the T transposition operation. Equation 6 is correct, that the transposition should be applied to K.
	- In short, sim(Q_i, K_j) is the similarity between two vectors, and should be a scalar. So $\phi(Q_i) \phi(K_j)^T$ should be (1xD) (Dx1) to generate 1x1 result, a scalar. 
	- Q is NxD, and $\phi(Q_i)$ and $Q_i$ is a row vector, 1xD.

