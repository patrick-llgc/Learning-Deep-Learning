# [Channel Pruning for Accelerating Very Deep Neural Networks](https://arxiv.org/pdf/1707.06168.pdf)

_May 2019_

tl;dr: Pruning filters by minimizing feature map reconstruction error. 

#### Overall impression
This paper is highly related to [pruning filters](pruning_filters.md) which uses L1 norm to prune filters. The paper is also highly influential in model compression field, with 224 citation as of 05/26/2019. 

The paper demonstrated that **[Pruning Filters](pruning_filters.md) (max response) with L1 norm is sometimes worse than random pruning.** It is argued that max response ignored correlation between different filters. Filters with large absolute weight may have strong correlation.

#### Key ideas
- This paper focuses on redundancy of feature maps, rather than filters themselves. Inference time for channel pruning, utilizing inter-channel redundancy.
- Minimization of reconstruction error is achieved in two steps: channel selection with Lasso and feature map reconstruction with linear least squares.

#### Technical details
- Training based methods integrate sparse constraints in the training process. This is more costly than inference based approaches. 
- The paper works on feature maps and thus is not a **data-free** method (needs sample data to generate feature maps). **Methods working directly on filters are data-free**.
- [Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm) is the sqrt of sum of L2 norm of all elements (maybe can be called "element-wise" L2?).

#### Notes
- Questions and notes on how to improve/revise the current work  

