# [Multi-object Monocular SLAM for Dynamic Environments](https://arxiv.org/abs/2002.03528)

_June 2020_

tl;dr: Multi object SLAM with keypoint. 

#### Overall impression
"Multibody mono SLAM" comes from "[multibody SfM](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/ozden10pami.pdf)", and means it not only tracks camera motion but also the motion of the dynamic rigid objects in the scene. 

Conventional vSLAM discards dynamic objects in the scene, multibody vSLAM explicitly estimates the pose of objects. 

#### Key ideas
- shape prior  + keypoint to lift 2D to 3D. 
- 3D depth estimation to ground plane points --> ?
- Mono 3DOD: Pose and shape estimation via 36 keypoints. 
	- Predicts keypoints with stacked hourglass network
	- recovers pose parameter and shape parameter (deformation from shape prior)

#### Technical details
- This paper claims that [cube SLAM](cube_slam.md) only estimates per-frame relative pose for each object and does not use it to construct a trajectory to avoid relative scale ambiguity --> Really?

#### Notes
- Questions and notes on how to improve/revise the current work  

