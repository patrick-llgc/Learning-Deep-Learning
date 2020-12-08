# [Centroid Voting: Object-Aware Centroid Voting for Monocular 3D Object Detection](https://arxiv.org/abs/2007.09836)

_December 2020_

tl;dr: Use bbox to guide depth prediction.

#### Overall impression
The paper is really a run-of-the-mill paper. The main idea is that instead of convolutional features to regress distance, use geometric prior to guide the distance prediction. The convolutional appearance features are only required to learn the residual.

#### Key ideas
- Based on two-stage object detector such as Faster-RCNN
- The paper uses two different modules to learn appearance
	- GPD (geometric projection distribution): predicts 3D center location's projection in 2D image
	- AAM (appearance attention map): 1x1 conv attention to address occlusion or inaccuracy in RPN


#### Technical details
- From Table IV, it seems that the fusion from the geometric branch did not help that much. However, from Fig. 6, it seem that without geometry the performance from appearance based method alone is not good at all.

#### Notes
- Maybe we can use this for distance prediction.

