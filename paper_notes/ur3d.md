# [UR3D: Distance-Normalized Unified Representation for Monocular 3D Object Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/6559_ECCV_2020_paper.php)

_August 2020_

tl;dr: Use distance-related feature transformation prior to ease the learning.

#### Overall impression
The paper used **depth normalization** for monocular 3D object detection. Similar ideas have been used in [Monoloco](monoloco.md) and [BS3D](bs3d.md). 

The paper has the good idea to learn the scale-invariant features and linearly scale them properly according to the FPN levels the regression heads are attached to. 

However the prediction tasks have known relationship according to scale, and we do NOT need to explicitly learn them. For example, bbox sizes are linearly related to scale, and depth scales inverse linearly to scale, both with a factor of 2 every FPN layer. The paper also seems to confuse the notion of depth (z) and distance (l2norm((x, y, z)).

The results are not SOTA as compared to [pseudo-lidar](pseudo_lidar.md) or [AM3D](am3d.md). It is further lagging behind contemporary work [PatchNet](patchnet.md).

[UR3D](ur3d.md) is largely based on the architecture of [FCOS](fcos.md). Similarly, [SMOKE](smoke.md) is based on [CenterNet](centernet.md).

#### Key ideas
- Features can be grouped into the three categories.
	- scale invariant tasks: 
		- object class
		- physical size
		- orientation
	- scale linear tasks. Each level of FPN regresses one scaling constant.
		- bbox
		- keypoint location
	- scale nonlinear tasks. Each level of FPN regresses one scaling constant.
		- depth prediction
- Uses [DORN](dorn.md) to generate a depth prediction patch to guide the learning of depth. Thus the network only needs to learn the residual values. 
- **Distance-guided NMS**: As distance or depth prediction is the key to accurate 3D object detection, the NMS is guided by distance prediction accuracy, not clf score alone.
	- Use depth conf * cls conf as sorting criteria, and used to weight average the depth value. 
- Fully convolutional cascaded point regression
	- 1st stage: regress the location of center point first
	- 2nd stage: use [deformable convnet](https://arxiv.org/abs/1703.06211) framework to pool all related points and predict the residual location offset.
- Postprocess to optimize 3D bbox according to predicted 9 keypoints and regressed physical size. 

#### Technical details
- Losses
	- [Wing loss](https://arxiv.org/abs/1711.06753) for distance, size and orientation estimation.
	- Smooth L1 loss for keypoint regression.
	- IoU loss for bbox regression

#### Notes
- Questions and notes on how to improve/revise the current work  

