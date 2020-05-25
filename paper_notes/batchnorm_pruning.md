# [BatchNorm Pruning: Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/abs/1802.00124)

_May 2020_

tl;dr: Similar idea to [Network Slimming](network_slimming.md) but with more details.

#### Overall impression
Two questions to answer:

- Can we set wt < thresh to zero. If so, under what constraints?
- Can we set a global thresh to diff layers?

Many previous works are norm-based pruning, which do not have solid theoretical foundation. One cannot assign different weights to the Lasso regularization to diff layers, as we can perform model reparameterization to reduce Lasso loss. In addition, in the presence of BN, any linear scaling of W will not change results.


#### Key ideas
- This paper (together with concurrent work of [Network Slimming](network_slimming.md)) focuses on sparsifying the gamma value in BN layer. 
	- gamma works on top of normalized random variable and thus comparable across layers.
	- The impact of gamma is independent across diff layers.
- A regularization term based on L1 of gamma is introduced, but scaled by a per layer factor $\lambda$. The global weight of the regularization term is $\rho$.
- ISTA (Iterative Shrinkage-Thresholding Algorithm) is better than gradient descent. 

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

