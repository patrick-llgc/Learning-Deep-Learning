# [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf) (ResNeXt)

_Feb 2019_

tl;dr: Introduce a new dimension *Cardinality* besides depth (#layers) and width (#channels) of CNN. Spliting the computation from width to cardinality leads to better performance. 

#### Overall impression
The paper exploits the multi-path (split-transform-merge) strategy, simplifies the design rules and introduces a new dimension. The better performnace than ResNet is a strong tesimony. This should be compared to other papers like Xception and mobileNet to see how they fare with each other.

#### Key ideas
- **Cardinality** is the size of the set of transformation. Increasing cardinality is more effective than going deeper or wider when increasing the capacity (FLOPs, #params).

- The aggregated transformation can be expressed as:
    $$
    y = x + \sum_{i=1}^C T_i(x)
    $$

- Split-transform-merge strategy of inception module

    > The solution space of inception module is a strict subspace of the solution space of a single larger layer (e.g, 5x5) operating on a higher-simentinal embedding. The split-transform-merge behavior is expected to approach the representational power of larger and dense layers, but at a considerbaly lower computiaonl complexity (and easier to train). 

- The authors argue that ResNet/ResNeXt should not be seen as ensemble of shallower networks as the paths are trained jointly, not independently.
- In the most simple instantiation of ResNeXt, cardinality can be implemented by **Group Convolution**. This reduces the computation from N x N to (N / C) ^2 * C = N x N / C. 

#### Technical details
- ResNeXt module has fixed input width and output width at 256 channels, and all widths are doubled each time when the feature map is subsampled by 2. Bottleneck width and cardinality are parameters controling the model capacity. ResNeXt with depth of 50, cardinality of 32 and bottleneck width of 4 are denoted as ResNeXt-50 (32x4d).

#### Notes
- Code in [pytorch](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnext.py) from mmdetection.

  

