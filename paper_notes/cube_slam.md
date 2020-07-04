# [CubeSLAM: Monocular 3D Object SLAM](https://arxiv.org/abs/1806.00557)

_June 2020_

tl;dr: Monocular SLAM with 3D MOD.

#### Overall impression
This is an extension to [orb-slam](orb_slam.md) by tracking higher level objects rather than key points alone. The name seems to be a play on [QuadricSLAM](https://arxiv.org/abs/1804.04011) <kbd>RAL 2017</kbd> where each object is represented by an ellipsoid.

It also has a way to perform single frame 3D object detection using 2D bbox and computer vision. Recent advances in DL based mono 3DOD can benefit this aspect. 

The SLAM part integrated the size and location of the cuboid into the factor graph optimization. Also it uses motion model to constrain possible movement of cuboids. This way the velocity is also part of the factor graph optimization. This is to be compared with the more flexible DL approach in [struct2depth](struct2depth.md).

Most existing SLAM methods assume the environment to be static or mostly static. Dynamic SLAM is the way to go for autonomous driving.

#### Key ideas
- Object detection + SLAM can benefit each other
	- Object provide geometry and scale constraints for BA, and depth initialization. It can also add to generalization capability to let orb slam work in low texture scenario. 
	- the mono3D results are optimized with BA, and constraint by a motion model
- Multiview object SLAM
	- decoupled approach: build SLAM point cloud then do detection 
	- coupled approach: object-level SLAM
- mono3D
	- Canny edge detector and vanishing point sampling
		- 3 VP and one vertices on top edge of 2d bbox.
	- Scoring function to score 3D proposals
		- distance error based on canny edges
		- angle alignment error based on line features and vanishing points
		- shape error based on object priors
- static object SLAM
	- Mono3D jointly optimize camera pose and object pose. 
	- BA formulation:
		- camera-object:
			- 3D: landmark object (tracked object) with current mono3D object prediction. 6 DoF pose + 3 DoF dimension
			- 2D: min bounding rect for 8 projected points, compared with the 2d detection bbox. --> we can use line features such as XPE or tireline too to formulate this reprojection error!
		- object point
			- for point associated with bbox, it should lie within the WHL cuboid.
		- camera point
			- Same as orb-slam. 
- point and object data association (static)
	- Feature point matching
	- if point are within the same bbox for 2 frames and are <1 m from cuboid center
	- Match two objects in diff frames if they have the most number of shared feature points and the num > 10. **Dynamic points are filtered out as they do not fulfill epipolar constraints**, so dynamic objects have very few points. Feature points in overlapping areas between bboxes are ignored. 
	- Static or not? Whether matched point on them satisfy epipolar constraints. 
- dynamic SLAM
	- motion model: nonholonomic vehicle model, linear velocity (yaw, steering angle as internal state)
	- dynamic point is anchored to the associated objects. The relative location of the point wrt the object is fixed. Thus dynamic point has the reproduction error. 
	- point and object association (dynamic)
		- KLT optical flow to track feature **points**. Object movement $\Delta T$, can be formulated as camera pose change equivalently, and can be solved by SVD. 
		- KLT tracking may still fail, thus dynamic **object** tracking is done with visual object tracking methods.
	- Dynamic object's speed profile can be estimated roughly within 1 m/s, even with a piecewise constant motion assumption. 
![](https://cdn-images-1.medium.com/max/1600/1*A0usFOJC6WEhr2VbJwlRtw.png)

#### Technical details
- Sampled based cuboid generation: 15 yaw angles
- only do mono3DOD on keyframes of orb-slam.
- Camera height provided to set scale
- Put less confidence for more distant cars.
- velocity stays the same for about 5 seconds. The model constraint is only applied to observations in the last 5 seconds. Piecewise constant function. 
- In some scenario with few feature points where orb slam cannot work, cube slam with object-camera can still work.
- without loop closure, orb slam works well only before first turning. Object SLAM can provide geometry constraints to reduce the scale drift to monoSLAM. Most existing methods are using the assumption of constant ground height to reduce scale drift. A combined method of using object constraints and ground plane height works the best.

#### Notes
- Decoupled approaches
	- [Visual-Inertial-Semantic Scene Representation for 3D Object Detection](https://arxiv.org/abs/1606.03968) <kbd>CVPR 2017</kbd>
- [code on github](https://github.com/shichaoy/cube_slam)
- [commented version](https://github.com/wuxiaolang/Cube_SLAM_wu)
- KLT tracking may fail for large pixel displacement, and visual object tracking is more robust. 

