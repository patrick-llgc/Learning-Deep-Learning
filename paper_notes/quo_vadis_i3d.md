# [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/pdf/1705.07750.pdf)

_Feb 2019_

tl;dr: Inflated 3D CNN (I3D) with weight bootstrapped from 2D imagenet pretrained weights.

#### Overall impression
One important drawback of 3D CNN is the lack of good initialization strategy and the lack of large datasets to pretrain its weight. This paper demonstrated that 2D weights pretrained on imagenet can be a good initialization strategy for 3D CNN as well. 

#### Key ideas
- The inflated 3D CNN has the same overall architecture as its 2D counterpart (such as 3D ResNet, 3D DenseNet, 3D VGG, etc). Each NxN filter is dilated to NxNxN. Note that the filter does not necessarily need to be cubic, and the temporal dimension can have different dimension than spatial ones.
- Bootstrapping 3D filters from 2D filters: The filter is duplicated N times along the time dimension, and rescaled by 1/N. The rescaling is critical to ensure the convolutional filter response to be the same.
- A symmetric receptive field is however not necessarily optimal when considering time, which should depend on frame rate and image dimension. 

#### Technical details
- Input size is 64x224x224.
- First two max pooling is 1x3x3, and the last average pooling is 2x7x7. All other operators are symmetric. 
- Fig. 4 shows that after training, the 64 conv1 filters are specialized to detect different patterns in different temporal dimension. This figure is quite convincing.

#### Notes
- For medical image analysis, if the volume is resampled to be isotropic, then we could use symmetric receptive field for the 3D filter.
- It would be great if 1) a large dataset in 3D medical domain is available, and 2) a set of diverse tasks are defined. These two are critical to generate effective pretrained weights in medical domain.
- The I3D source code is written in Sonnet. A pytorch implementation is [here](https://github.com/hassony2/inflated_convnets_pytorch).
- There are several other papers that also experimented with initialization schemes for 3D CNN with 2D CNN weights. 
	- [Detect-and-Track: Efficient Pose Estimation in Videos](https://arxiv.org/pdf/1712.09184.pdf): center initialization is better than mean initialization with a 3D mask RCNN backbone.
	- [Initialization Strategies of Spatio-Temporal Convolutional Neural Networks](https://arxiv.org/pdf/1503.07274.pdf): center initialization is indeed better than mean initialization. However there is an even better way to overshoot the center slice and compensate by negative weight in the neighboring slices.