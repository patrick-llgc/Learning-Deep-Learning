# [Multi-object Monocular SLAM for Dynamic Environments](https://arxiv.org/abs/2002.03528)

_June 2020_

tl;dr: Multi object SLAM with keypoint. 

#### Overall impression
"Multibody mono SLAM" comes from "[multibody SfM](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/ozden10pami.pdf)", and means it not only tracks camera motion but also the motion of the dynamic rigid objects in the scene. 

Conventional vSLAM discards dynamic objects in the scene, multibody vSLAM explicitly estimates the pose of objects. 

#### Key ideas
- shape prior  + keypoint to lift 2D to 3D. 
	- k=36 ordered keypoints. B basis vector is $V \in R^{3k \times B}$. The deformation coefficents be $\Lambda \in R^B$.
	- Lifting 2D to 3D by minimizing reprojection error to get 6 DoF pose and shape parameter $\Lambda$. This is similar to [RoI-10D](roi10d.md)
- metric ego-motion odometry: ground plane constraint orb-slam.
	- Scale factor: 3D depth estimation to ground plane points, by IPM and semantic segmentation of the ground within 12 meters. 
	- odometry up to scale: Orb-slam.
- Mono 3DOD: Pose and shape estimation via 36 keypoints. 
	- Predicts keypoints with stacked hourglass network
	- recovers pose parameter and shape parameter (deformation from shape prior)
- Multi-object pose graph optimizer
	- node in g2o graph: *estimates*
	- edge in g2o graph: *measurement*
	- use **cyclic consistency** as constraint. This is equivalent to **minimization of reprojection error**.
	- Camera-camera edge is obtained via metric-scale odometry
	- camera-vehicle edge is obtained via single frame 3DOD
	- vehicle-vehicle edge is obtained via two different 3D depth estimation method (IPM vs 2d-to-3d lifting). This avoided explicit motion model. 

#### Technical details
- This paper claims that [cube SLAM](cube_slam.md) only estimates per-frame relative pose for each object and does not use it to construct a trajectory to avoid relative scale ambiguity --> Really?

#### Notes
- Questions and notes on how to improve/revise the current work  

