# [LcP: Layer-compensated Pruning for Resource-constrained Convolutional Neural Networks](https://arxiv.org/pdf/1810.00518.pdf)

(NIPS 2018 [talk](https://sites.google.com/view/nips-2018-on-device-ml/schedule?authuser=0) for ML on device)

_May 2019_

tl;dr: Layer-wise pruning, but with layer-compensated loss. 

#### Overall impression
Previous method approximates the pruning loss increase with the L1 or L2 of the pruned filter. This is not true. LcP first approximates the layer-wise error compensation and then uses naive pruning (global greedy pruning algorithms) to prune network.

#### Key ideas
- Two problems in pruning the network: how many to prune and which to prune. The first is also named layer scheduling. 
- Naive pruning algorithm: global iterative pruning without layer scheduling.
- Two approximation in prior art of multi-filter pruning:
	- Approximate loss change with a ranking metric (the paper addresses this issue)
	- Approximate the effect of multiple filter pruning with addition of single layer pruning.
- The paper assumes that the approximation error to be identical for filters in the same layer. Therefore only L latent variables $\beta_l, l=1, ..., L$ need to be approximated. 

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

