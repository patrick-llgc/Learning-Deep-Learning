# [KP3D: Self-Supervised 3D Keypoint Learning for Ego-motion Estimation](https://arxiv.org/abs/1912.03426)

_March 2020_

tl;dr: Predict keypoints and depth from videos simultaneously and in a unsupervised fashion.

#### Overall impression
This paper is based on two streams of unsupervised research based on video. The first is depth estimation starting from [sfm Learner](sfm_learner.md), [depth in the wild](mono_depth_video_in_the_wild.md) and [scale-consistent sfm Learner](sc_sfm_learner.md), and the second is the self-supervised keypoint learning starting from [superpoint](superpoint.md), [unsuperpoint](unsuperpoint.md) and [unsuperpoint with outlier rejection](kp2d.md).

The two major enablers of this research is [scale-consistent sfm Learner](sc_sfm_learner.md) and [unsuperpoint](unsuperpoint.md).

The main idea seems to be using sparse matched keypoint pairs to perform more accurate (relative) pose estimation. Previously the ego motion is directly regressed from This leads to much better ego motion estimation.

#### Key ideas
- Some notation convention:
	- $p_t \in I_t$ keypoints in target image and $p_s \in I_s$ keypoints in source image
	- $p_t^{MV} \in I_s$ matched keypoints of $p_t$ in source image based on descriptor space. Based on the pair of $p_t \leftrightarrow p_t^{MV}$ we can compute the associated ego motion $x_{t \rightarrow s}$. Descriptor loss is on this. 
	- $p_t^* \in I_s$ warped $p_t$ in source image ($\hat{p_t}$ in KP2D ). Sparse keypoint location loss is between $p_t^{MV}$ and $p_t^*$.
	- Once $x_{t \rightarrow s}$ is known, dense Photometric loss and sparse keypoint location loss are formulated.
- In the whole pipeline, calculating $p_t^*$ is the hardest. In Homography Adaptation $p_t^*$ can be calculated trivially, but in multi-view adaptation this is hard and need to project to 3D via $\pi^{-1}(R|t)$.
- Instead of using CNN directly for pose estimation (PoseNet in [sfm Learner](sfm_learner.md)), KP3D uses matched keypoint to do pose estimation, and this could be the key to the better performance ([superpoint](superpoint.md) and [unsuperpoint](unsuperpoint.md) are known to yield very good HA, homography accuracy).
- Added depth consistency, as the depth is scale-ambiguous. It is critical for ego-motion estimation. A sparse loss between $p_t$ and $p_t^{MV}$ is used. 


#### Technical details
- Training process:
	- Pretraining keypoint detector (similar to [KP2D](kp2d.md)).
	- KeypointNet and DepthNet both imageNet pretrained ResNet18. 
	- Changing backbone from VGG in [KP2D](kp2d) to ResNet18 in [KP3D](kp3d.md) improves performance.

#### Notes


