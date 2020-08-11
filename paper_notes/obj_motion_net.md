# [ObjMotionNet: Self-supervised Object Motion and Depth Estimation from Video](https://arxiv.org/abs/1912.04250)

_August 2020_

tl;dr: Train a PoseNet-like ObjMotionNet to predict 6 DoF pose change of each object in monodepth.

#### Overall impression
The idea of predicting object level motion is similar to [Struct2Depth](struct2depth.md). It focuses on the depth of the foreground objects, similar to [ForeSeE](foresee_mono3dod.md). It predicts 6 DoF object pose change, similar to [VelocityNet](velocity_net.md).

The paper only deals with objects with rigid motion, such as cars and trucks, and does not deal with non-rigidly moving objects such as pedestrians.

#### Key ideas
- Segment the temporal image sequence by the instance aligned mask. A ObjMotionNet is trained jointly with depthNet to predict the 6 DoF pose of each object. 
- Losses:
	- Photometric reconstruction loss
	- Left-right photometric loss (for stereo only)
	- Edge-aware smoothness loss
	- Geometric constraint loss
		- Mean depth inside object mask should match that of the ObjMotionNet. 
- Multi-stage training
	- DepthNet and left right consistent loss. No photometric loss to avoid contamination from moving object. --> This can be only afforded by a stereo system. Maybe use [SGDepth](sgdepth.md) to filter out all objects in this step.
	- Train only ObjMotionNet and DepthNet.
	- Fix DepthNet and finetune ObjMotionNet on test set.


#### Technical details
- Overall monodepth2 is still better, but the fg depth performance is best. This is similar to [ForeSeE](foresee_mono3dod.md).

#### Notes
- Questions and notes on how to improve/revise the current work  

