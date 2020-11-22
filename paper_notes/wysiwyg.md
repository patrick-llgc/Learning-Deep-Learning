# [What You See is What You Get: Exploiting Visibility for 3D Object Detection](https://arxiv.org/abs/1912.04986)

_June 2020_

tl;dr: Visibility augmented deep voxel representation, with occupancy grid feature map.

#### Overall impression
Representing point cloud as xyz fundamentally destroys the difference of free space and uncertainty space (both contains no lidar points). Convolution cannot differentiate such difference based on such a representation.

WYSIWYG adds an occupancy grid feature map to the existing feature map, very much like [coord conv](coord_conv.md) and [cam conv](cam_conv.md).

#### Key ideas
- Online occupancy mapping 
	- Following OctMap. Fast ray casting via voxel traversal --> visibility volume
	- Integration over time with Bayesian Filtering,
	- Uses the same discretization as BEV map. Voxel size 0.25x0.25x0.25 m^3. 
	- FoV [-50, 50] x [-50, 50] x [-5, 3] --> 400x400x32 occupancy grid. 
- Visibility over multiple ldiar sweeps: bayesian filtering
- data augmentation
	- Naive data augmentation: copy and paste rare objects such as buses. But it violates visibility rules. 
	- Visibility aware data augmentation
	- Drilling is better than culling as culling removes too many invalid cases, especially for big cars (hard to place them in a place where it does not violate visibility)
- The idea of ray casting is widely used in generating lidar simulated data, such as [lidar sim](lidar_sim.md).


#### Technical details
- Reconstructed point cloud and measured point cloud (lidar sweeps) are different in that measured point cloud also includes the visibility information.
	- Lidar sweeps are 2.5D
	- True 3D point cloud data are, for example, sampled from mesh models
- Two crucial innovation in training lidar object detector recently: **Object augmentation and temporal aggregation**. They are first proposed in SECOND, and then used in all SOTA methods such as [pointPillars](point_pillars.md). The temporal aggregated lidar frames should be motion compensated.
- Training follows [One cycle policy](https://sgugger.github.io/the-1cycle-policy.html). This is also used in [centerpoint](centerpoint.md) and [CBGS](cbgs.md).

#### Notes
- [Review about data representation](https://zhuanlan.zhihu.com/p/143670859)
- [Talk at CVPR 2020](https://www.youtube.com/watch?v=497OF-otY2k)
- [github code](https://github.com/peiyunh/WYSIWYG). The codebase is based on SECOND, similar to [pointPillars](point_pillars.md).
- [Occupancy grid maps, lecture by Cyrill Stachniss](https://www.youtube.com/watch?v=v-Rm9TUG9LA)

