# [Stereo Vision-based Semantic 3D Object and Ego-motion Tracking for Autonomous Driving](https://arxiv.org/abs/1807.02062)

_June 2020_

tl;dr: object SLAM that uses 3d mod proposals from each frame. 

#### Overall impression
The demo is quite impressive. Almost as good as [cube slam](cube_slam.md). The 3D object proposal step is quite simple yet effective for cars like sedans with strong shape priors, even simpler than the [deep3dbox](deep3dbox.md) method.

Many insightful comments from the paper:

> End to end 3D regression need lots of training data and require heavy workload to precisely label all the object bboxes in 3D. Instance 3D detection produces frame-independent results, which are not consistent enough for continuous perception in autonomous driving. 

> Purely depending on instance 2D bbox limits its performance in predicting pose for truncated object. 

The paper proposed a novel object bundle adjustment (BA). The method can track 3D objects and recover he dynamic sparse point cloud with **instance accuracy and temporal consistency**.

The method can track the object continuously even for the extremely truncated case where object pose is hard for instance inference. 

#### Key ideas
- 3d bbox proposal:
	- 2D bbox + 8 viewpoint classification + shape dimension prior
	- Assumption: reprojection of 3D bbox will tightly fit the 3d bbox. --> This assumption is not exactly true. But for most autonomous driving cases (horizontal or slightly looking down) the assumption holds true quite well. 
- Ego motion tracking
	- Filter out dynamic objects and do slam to get pose. Similar to orb-slam.
- Object tracking: 4 residual loss terms
	- feature reprojection (feature points on objects has fixed coordinates in object frame)
	- 2D bbox consistency (projection of 3D tracked object should fit into 2D measurement)
	- 3D pose consistency (temporally predicted pose vs inferred measurement from frame). A kinematics motion model to ensure consistent orientation and motion estimation. It involves vehicle dim, speed, steer angle.
	- object dimension consistency
- Dynamic Point cloud alignment (for stereo only)
	- After getting pose, align it with feature point cloud with accurate depth information from stereo.

#### Technical details
- Dimension of objects are invariant!
- Adding feature point tracking helps with easy cases (large, untruncated cars), while adding dynamic point cloud alignment helps with hard cases (faraway, truncated cars)

#### Notes
- [Demo on youtube](https://www.youtube.com/watch?v=nE2XtCvPEDk)

