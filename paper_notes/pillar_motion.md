# [PillarMotion: Self-Supervised Pillar Motion Learning for Autonomous Driving](https://arxiv.org/abs/2104.08683)

_August 2021_

tl;dr: Self-supervised pillar motion learning.

#### Overall impression
This paper is benchmarked against [MotionNet](motionnet.md). The backbone follows that of [MotionNet](motionnet.md), but instead of using bbox as annotation, it uses the optical flow information from camera image as supervision signal.

Pillar Motion can be used for better tracking, and planning in case of an unknown object.

#### Key ideas
- Assumes all points within a pillar has the same motion --> this makes the smoothness loss insignificant
- Loss
	- Structural consistency between the transformed point cloud $\tilde P$ and real point cloud P. The transformation is by motion vector M.
		- This could be noisy in static region due to point cloud sparsity and occlusion.
	- Cross sensor motion regularization.
		- Factor out the ego motion out from optical flow and get optical flow from true object motion $F_{obj}^t$
		- Optical flow caused by ego motion: $F_{ego}^t(u, v) = KT_{L\rightarrow C} T_{t \rightarrow t+1} P^t - KT_{L\rightarrow C}  P^t$ 
		- Project motion vector into camera image and get optical flow $F^t (u,v)$
		- L1 loss $|F^t(u,v) - F_{obj}^t(u,v)|_1$
	- Probabilistic Motion masking
		- the probability of being static is estimated by $F_{obj}^t(u,v)$. Then reproject back to the lidar point cloud. 
		- The static region's loss is downweighted.
	- Smoothness loss of motion: not that important due to the assumption that all points in a pillar has the same motion
- Pretraining can be paired with fine-tuning on labeled images (like that used in [MotionNet](motionnet.md)) can achieve the same level performance with much less (~20%) labeled data.


#### Technical details
- Estimation scene flow from images and enforce consistency with lidar points in 3D is still very hard. Predicting optical flow and enforce consistency with the projection of lidar in 2D is more practical.

#### Notes
- Code to be released in [github](https://github.com/qcraftai/pillar-motion).

