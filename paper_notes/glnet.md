# [GLNet: Self-supervised Learning with Geometric Constraints in Monocular Video: Connecting Flow, Depth, and Camera](https://arxiv.org/abs/1907.05820)

_July 2020_

tl;dr: Combine monodepth with optical flow with geometric and photometric losses.

#### Overall impression
The paper proposes two online refinement strategies, one finetuning the model and one finetuning the image. --> cf [Struct2Depth](struc2depth.md) and [Consistent video depth](consistent_video_depth.md).

It also predicts intrinsics for videos in the wild. --> cf [Depth from Videos in the Wild](learnk.md).

The paper has several interesting ideas, but there are some conflicts as well. The main issue is that it uses FlowNet to handle dynamic regions but it still enforces epipolar constraints on the optical flow. Also it does not handle depth of the dynamic regions well. 

Geometric constraints are more powerful than photometric constraints. 

#### Key ideas
- Architecture: PoseNet + DepthNet + FlowNet. 
- Terminology: 2D image: p and p'; 3D space: x and x'.
- Adaptive photometric loss:
	- Pixels in two images are either due to ego motion, or secondary or non-rigidly moving objects. For scene structure not explainable by global rigid motion, one can rely on the more flexible optical flow.
	- L = min(ego motion photometric loss, optical flow photometric loss)
	- Q: if the FlowNet does a very good job, then depthNet is not well trained?
- Multiview 3D structure consistency:
	- Projected points p and p' back to 3D, x and x' should agree with each other. It is a 3D loss.
- Epipolar constraint loss for optical flow
	- p and p' should be connected by the essential matrix expressed by the pose (by E = t^R).
- Online refinement strategies.
	- Parameter finetuning
	- Output finetuning: faster as the number of pixels is smaller than number of params.

#### Technical details
- With multiview loss, after refinement the loss drops to the lowest. 
- Each training sample is a snippet consisting of three consecutive frames.

#### Notes
- FlowNet is used to model moving objects. However the epipolar constraint breaks for moving objects. Maybe a mask would work better?
- The depth of moving object is still not well handled. 

