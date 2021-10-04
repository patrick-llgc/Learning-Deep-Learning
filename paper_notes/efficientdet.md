# [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)

_September 2021_

tl;dr: BiFPN and multidimensional scaling of object detection.

#### Overall impression
This paper follows up on the work of [EfficientNet](efficientnet.md). The FPN neck essentially is a multi-scale feature fusion that aims to find a transformation that can effectively aggregate different features and output a list of new features.

#### Key ideas
- BiFPN (bidirectional FPN) (<-- PANet <-- FPN)
	- [PANet](panet.md) to introduce bottom up pathway again.
	- **Remove nodes** from PANet that has only has one input edge.
	- **Add skip connection** from original input to the output node if they are at the same level
	- **Repeat** blocks of the above BiFPN block.
- Weighted feature fusion
	- Baseline is to resize and sum up. Each feature may have different weight contribution (feature level attention).
	- Softmax works, but a linear weighting normalization may work as well.
- Multidimensional/compound scaling up is more effective than single dimension scaling. Resolution, depth and width.

#### Technical details
- [NAS-FPN](nas_fpn.md) has repeated irregular blocks.
- Simply repeating FNP blocks will not lead to much benefit. Repeating PANet blocks will be better, and repeated BiFPN yields similar results but with much less computation.
- This still needs object assignemtns, like [RetinaNet](retinanet.md).

#### Notes
- [Github](https://github.com/google/automl)

