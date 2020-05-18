# [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) 

_May 2019_

tl;dr: Pruning filters instead of sparsely pruning weight for efficient inference.

#### Overall impression
This is one of the first paper in model pruning (with ~400 citation as of 05/21/2019, and ~800 as of 05/17/2020).

#### Key ideas
- Magnitude based pruning with L1 norm (the authors commented that L2 norm yields similar results).
- Conventional pruning leads to irregular sparsity in pruned network, and requires  sparse conv libraries and special hardwares.
- **Pruning parameters does not necessarily reduce the computation time** since the majority of the parameters removed are from the fully connected layers where the computation cost is low. (FC layer in VGG-16 occupy 90% of parameters but less than 1% of the computation).
- **Reducing FLOPs does not necessarily reduce energy cost**. 1 access to memory actually is ~1000 more energy consuming than ADD. ([source](https://youtu.be/eZdOkDtYMoo?t=223)). 16 FP Mult takes 1/4 of energy of 32 FP Mult.
- The paper proposes to prune multiple filters at once and retrain once. (as opposed to conventional pruning of one filter at a time and retrain after pruning each filter).
- Pruning based on weights is data free, whereas pruning based on activation map needs test images. 
- **Pruning m filters reduces $m/n_{i+1}$ of the computation cost for both layer i and i+1.** Because it not only removed the filters i+1 but also feature map, which is the input of the i+1 layer.

#### Technical details
- Each filter is $n_i$ filters, thus of size $n_i \times k \times k$, where $n_i$ is the channel of input data. L1 ranking score is $s_j = \sum_{l=1}^{n_i} \sum_{k \times k} |K_l|$. The smaller the score, the less important the filter is.
- No noticeable difference in L1 or L2 norm in pruning filters, as important filters tend to have large values in both measures. 
- Retrain: for pruning-resilient layers, pruning away large number of filters and retrain once yields good results.

#### Notes
- A filter and a 2D kernel are two diff concepts. One filter can contain num_channel 2D kernels. 
- Parameters and computation are not necessarily correlated.
- Pruning filters will leads to computation cost savings in two nearby layers.

