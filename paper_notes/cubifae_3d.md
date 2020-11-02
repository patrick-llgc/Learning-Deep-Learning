# [CubifAE-3D: Monocular Camera Space Cubification on Autonomous Vehicles for Auto-Encoder based 3D Object Detection](https://arxiv.org/abs/2006.04080)

_October 2020_

tl;dr: Use depth pretraining with AE on synthetic data to help Mono3D.

#### Overall impression
The idea of using depth pretraining for mono3D is similar to [Geometric pretraining](geometric_pretraining.md) for monoDepth. The pretraining can be done with synthetic dataset. Maybe the self-supervised pretraining can also work. 

The idea of cubifying 3D space is densely sample 3D space, and is similar to the idea of 3D anchors in [M3D-RPN](m3d_rpn.md). The idea of pool 2D image features into 3D voxels resembles that of [OFT](oft.md).

The most contribution to this work seems to be the improvement of 3D detection for far away objects. It in a way eliminated the depth dependency of prediction errors. --> How is this done?

The GT assignment looks interesting as it predicts up to 10 cars in each cuboid and they are sorted by increasing depth. Anchor is a way to implicitly sorting the prediction and GT. [DETR](detr.md) is quite radical in eliminating the ordering and sorting of GT and prediction altogether and replace with a Hungarian matching loss. 

#### Key ideas
- Pretraining monoDepth with synthetic dataset.
	- U-Net like structure, with MSE error and edge aware smoothing error. 
	- The pretrained weights from depth encoder is frozen, and the depth decoder is discarded.
	- The networks learns the linear relationships between relative focal lengths and relative depth scales between simulation and real datasets, regardless of differences in camera intrinsics. 
- Cubify the entire place into (2x2)x5 depth bin x 10 objects per bin. 
	- predict 10 object per cuboid, in increasing order of z-depth from ego vehicle. 
	- Each quadrant of the image with depth limit is one cuboid. 
- Loss
	- whl loss: L2 of sqrt diff
	- xyz loss: L2 loss
	- orientation loss: L2 loss
	- conf loss: L2 loss
	- iou loss: introducing this loss speeds up the stabilizes the training process.
- Classifier: a separate classifier network to crop the patches out with the predicted whl to predict vehicle class. 

#### Technical details
- Photometric data aug: RGB to HSV first, then adjust S and V. 
- Geometric data aug: projective data aug, not done yet as it requires determination of new object pose.

#### Notes
- Q: is it possible to use single image for depth pretraining? The idea would be similar to [Virtual Cam/Movi-3D](movi_3d.md) and [Cam Conv](cam_conv.md), by augmenting single image and its depth value. 

