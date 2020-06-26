# [What You See is What You Get: Exploiting Visibility for 3D Object Detection](https://arxiv.org/abs/1912.04986)

_June 2020_

tl;dr: Visibility augmented deep voxel representation, with occupancy grid feature map.

#### Overall impression
Representing point cloud as xyz fundamentally destroys the difference of free space and uncertainty space (both contains no lidar points). Convolution cannot differentiate such difference based on such a representation.

WYSIWYG adds a occupancy grid feature map to the existing feature map, very much like [coord conv](coord_conv.md) and [cam conv](cam_conv.md).

#### Key ideas
- Fast ray casting via voxel traversal --> visibility volume 
- Visibility over multiple ldiar sweeps: bayesian filtering
- data augmentation
	- Naive data augmentation: copy and paste rare objects such as buses. But it violates visibility rules. 
	- Visibility aware data augmentation
- The idea of ray casting is widely used in generating lidar simulated data, such as [lidar sim](lidar_sim.md).


#### Technical details

#### Notes
- [Review about data representation](https://zhuanlan.zhihu.com/p/143670859)
- [Talk at CVPR 2020](https://www.youtube.com/watch?v=497OF-otY2k)
- [github code](https://github.com/peiyunh/WYSIWYG). The codebase is based on SECOND, similar to [pointPillars](point_pillars.md).

