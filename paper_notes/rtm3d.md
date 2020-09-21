# [RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving](https://arxiv.org/abs/2001.03343)

_January 2020_

tl;dr: CenterNet-based method to directly detect 2d projection of cuboid vertices.

#### Overall impression
This paper uses virtual keypoint and use CenterNet to directly detect the 2d projection of all 8 cuboid vertices + cuboid center. The paper also directly regresses distance, orientation, size. Instead of using these values to form cuboid directly, these values are used as initial value to initialize the offline optimizer to generate 3D bbox.

The predicted keypoints are very noisy. However after optimization it is possible to regress to stable 3d bbox.

The architecture is easy to implement. The post-processing algorithm seems to be quite heavy by solving a multivariate equations can be solved via the Gauss-Newton or Levenberg-Marquardt algorithm in the g2o library. need more investigation. Natively they can be solved by a pseudo-inverse algorithm of an overdetermined linear system. --> this is improved by later work of [KM3D-Net](km3d_net.md).

#### Key ideas
- [CenterNet](centernet.md)
	- main center: 2d bbox center (also within image boundary)
	- 9 vertices 
	- 18 local offset (Q: what is the order? redundant info from 9 vertices?)
	- optional:
		- vertices offset (to counter discretization error)
		- local yaw
		- size
		- distance
- Bbox estimate:
	- Reprojection error in 2D
	- size error with prior (CNN regressed size)
	- Rotation error with prior (CNN regressed angle)

#### Technical details
- Why 8 bits for orientation estimation?

#### Notes
- Can we work the post-processing step into neural network as well? --> put these to a second step, and do not back-propagate the loss.

