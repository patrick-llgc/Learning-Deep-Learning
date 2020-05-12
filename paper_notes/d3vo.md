# [D3VO: Deep Depth, Deep Pose and Deep Uncertainty for Monocular Visual Odometry](https://arxiv.org/abs/2003.01060)

_May 2020_

tl;dr: Add depth estimation and brightness transformation to [Monodepth2](monodepth2.md). The paper incorporate the deep predicted pose into regularization into a backend. 

#### Overall impression
Monocular VO suffers from scale drift and low robustness. The pose from PoseNet in [sfm learner](sfm_learner.md) and [Monodepth2](monodepth2.md) are robust, but they are not as accurate as geometry based methods. This paper explores on how to combine geometric approach to deep learning approach (aka "hybrid"). 

Hybrid methods combines deep learning with geometry based methods: 

- [D3VO](d3vo.md) uses the pose and depth prediction in the backend of [DSO](dso.md)
- [DF-VO](df_vo.md) uses both optical flow and depth inside a robust essential matrix estimation loop.
- [KP3D](kp3d.md) also uses [DSO](dso.md) as backend.

As repeated demonstrated as with other hybrid methods, D3VO beats all other end to end methods by a large margin.

VO lacks robustness for low texture area and fast movement. VIO is more robust, but IMUs cannot deliver the metric scale in constant velocity.

Both [KP3D](kp3d.md) and [D3VO](d3vo.md) uses DSO as backned, and KP3D reaches on par performance with DVSO, while D3VO beats DVSO and even achieves comparable to stereo/lidar methods on KITTI odometry. 

#### Key ideas
- Predicts a **photometric uncertainty map** to capture regions that may violate the brightness constancy assumption (non-lambertian surface, moving objets). This is related to the **flow consistency map** in [DF-VO](df_vo.md) and **explainability mask** in [SfM-learner](sfm_learner.md).
- The paper also predicts a brightness transformation parameter (linear scaling) which is critical for some dataset (KITTI is largely stable). 
- D3VO backend is based on [DSO](dso.md). 
	- The virtual stereo term optimizes the estimated depth from VO to be consistent with the depth predicted by the proposed deep network.
	- The predicted pose is integrated in a similar fashion to with IMU with a Gaussian model, as a regularizer in the energy function.

#### Technical details
- Uses both left/right stereo consistency and temporal consistency. Quaduplets ($I_t$, $I_t^s$, $I_{t-1}$, $I_{t+1}$)
- Multiscale loss
- 40k training quadruplets.
- Uncertainty boosted performance on KITTI the most. 

#### Notes
- Questions and notes on how to improve/revise the current work  

