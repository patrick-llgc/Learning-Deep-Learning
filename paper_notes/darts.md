# [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)

_August 2021_

tl;dr: Differentiable optimization of neural architecture.

#### Overall impression
Previous methods uses evolution or reinforcement learning over a discrete and non-differentiable search space, where a large number of architecture evaluation is required. DARTS is based on continuous relaxation of the arch representation. 

DARTS does not involve training a controller based on sparse validation set  score (treated as the reward or fitness) with RL or evolution. DARTS directly optimizes the parameters controlling the architecture via gradient descent on the validation dataset.

The paper is very similar in idea to [FBNet](fbnet.md). 

#### Key ideas
- The categorical choice of a particular operation is relaxed to a softmax over all possible operations. This is even simpler than the Gumbel softmax in [FBNet](fbnet.md).
- Bi-level optimization
	- The optimization happens in an alternating fashion, one step optimizing w on the training set and one step optimizing a on the validation set.
	- This trick is widely used in metalearning (MAML).
- $\min_a L_{val}(w^*(a), a)$, s.t. $w^*(a) = \arg\min_w L_{train}(w, a)$
	- First order approximation is that w = w* and this is equivalent to the simple heuristic of optimizing the validation loss by assuming the current w is the optimal w* (trained until convergence). This leaves inferior results to second order approximation.

#### Technical details
- RL-based NAS or evolution based NAS all require 2000-3500 GPU days. 
- This is in a sense comparable with gradient-based hyper-parameter (such as LR) optimization.

#### Notes
- [Review on 知乎](https://zhuanlan.zhihu.com/p/156832334). Formula deduction can be viewed in details [here](https://zhuanlan.zhihu.com/p/73037439).

