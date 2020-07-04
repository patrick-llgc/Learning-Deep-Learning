# [ClusterVO: Clustering Moving Instances and Estimating Visual Odometry for Self and Surroundings](https://arxiv.org/abs/2003.12980)

_June 2020_

tl;dr: General dynamic slam that can detect and track 3D MOD.

#### Overall impression
The paper is similar in function to [Cube SLAM](cube_slam.md) that tracks objects and us it to increase the robustness of SLAM, and also is able to handle dynamic scenes. It is more flexible in the sense that the detection is based on **point cloud of landmarks on the cluster**. In comparison, [Cube SLAM](cube_slam.md) models each object as a 3d cuboid, and [QuadricSLAM](https://arxiv.org/abs/1804.04011) models each object as an ellipsoid.

The paper is based on a stereo system but the performance of 3DOD is even worse than [mono3D](mono3d.md). This is due to the way clusterVO generates 3D bounding box through tracked cluster, which may not subtend the whole size of the object. 

Terminology:

- cluster: cluster of point landmarks, rigid body, 3d objects. Raw measurement is bbox. 
- landmark: tracked feature points. 
- $q$: cluster or object, q=0 is the background (static)
- $q^i$: association of landmark i to cluster q.


#### Key ideas
- Based on 2D landmarks and sparse landmarks
- Multi-level Probabilistic association
	- feature points k --> landmark i
	- bbox m --> cluster q
	- landmark i --> cluster q
		- heterogeneous CRF with efficient inference
		- unary energy:
			- 2d bbox constraint: point should lie in bbox
			- 3d bbox constraints
			- how the trajectory of q can explain the observation of landmark
		- pairwise energy
			- encourages label smoothness: two close enough landmark should belong to the same cluster.
	- Hungarian matching (Kuhn-Munkres algorithm) to match clustering results from previous results (assigning index)
- State estimation
	- Double track frame management: temporal track $T_t$ (15) and spatial track $T_s$ (5). 
	- last frame need to be **marginalized** to save memory
	- static scene (q=0): BA with marginalization term. Optimize both on spatial and temporal tracks. 
	- dynamic cluster (q!=0): white noise acceleration. Only optimize more recent temporal track.


#### Technical details
- object detection with yolo v3.
- Motion model: acceleration is queried from a Gaussian random process. This penalizes change in velocity over time and smooth cluster motion trajectory which would otherwise be noisy. <-- this is to be compared with the piecewise constant velocity in [cube slam](cube_slam.md)
- The center and size are found by using 30th and 70th percentile of cluster landmark point cloud. This may be the reason of the low KPI as compared to mono3D. 
- Object yaw: this is likely determined by motion.
- [semantic 3d slam](semantic_3d_slam.md) uses batch multibody sfm and has a highly nonlinear factor graph optimization. The solution is not trivial.

#### Notes
- [demo on youtube](https://www.youtube.com/watch?v=paK-WCQpX-Y)
- the framework is not suitable for specialized autonomous driving use, especially for improved mono3D. The determination of size is based on the point cloud of landmarks, which may not reflect the real size of cars. The orientation seems to be determined using speed, and cannot be very accurate for still or object with small speed.
- What is marginalization? Shurr Complement?

