# [GPP: Ground Plane Polling for 6DoF Pose Estimation of Objects on the Road](https://arxiv.org/abs/1811.06666)

_November 2019_

tl;dr: Regress tireline and height and project to the best ground plane near the car.

#### Overall impression
GPP generates artificial 2D landmarks with 3D bbox annotation. It purposefully predicts more attributes than needed to estimate 3D bbox (over-determined) and uses these predictions to from **maximum consensus set** of attributes, in a manner similar to RANSAC, making it more robust to outliers. 

#### Key ideas
- Viewpoint (orientation) classes: 4 x 2. Depending on whether the central edge is on the left half or right half of 2D bbox (if local yaw is beyond 20 deg in a typical car).
- **Ground Plane Polling**: 
	- Given a plane candidate, get the projection of three tirelines. Form a virtual vertical backplane edge
	- Find the nearest point on the backprojected ray of backplane edge top point to the virtual edge (in practice they do not intersect).
	- 4 3D points form 6 edges pairs. The residual error of the 6 edges and real 3D length
	- The best fit plane minimizes the residual loss
	- Directly enforcing orthogonality led to most probable plane being discarded
	- Discard the tireline corresponding to width of the car (only using side tireline) to enforce orthoganality
	- Reconstruct the 3D bbox in a layer

#### Technical details
- RetinaNet backbone, classify into 8*K classes, 8 being the orientation class.
- Using RANSAC to create 22k ground plane candidates based on KITTI. This is with tight constraint (t = 2 cm) and very high probability of success (p = 0.999). In experiment, 10K planes are used.
- The plane is denoted by 4 numbers (as it is 4 DoF).
- Deep3Dbox cannot handle closeby objects well as the error goes up with very close distance.

#### Notes
- Maybe we can enforce all object within the image are on the ground. to make better prediction.
- The 2D/3D tight constraint looks invalid based on Fig. 5. Maybe not for closeby cars.
