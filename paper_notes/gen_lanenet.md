# [Gen-LaneNet: A Generalized and Scalable Approach for 3D Lane Detection](https://arxiv.org/abs/2003.10656)

_March 2020_

tl;dr: A two-stage method for 3D LLD and a new open dataset.

#### Overall impression
The paper pointed out one major drawback of [3D LaneNet](3d_lanenet.md) that the top view projection does not align with the IPM image feature. The proposed "virtual top view" projection will align with IPM. This essentially changes the regression target (not the "anchor representation" the paper argues.)

Concretely, an uphill 3D lanes with zero yaw will provide parallel target in the real top view (x, y, 0) but will have diverging target in virtual top view  ($\bar{x}$, $\bar{y}$, $0$). Similarly it will have converging target in virtual top view. This will align with the IPM'ed image features. 

In [3D LaneNet](3d_lanenet.md) there is no guarantee that the projection of prediction matches the image and lacks the 2D-3D consistency. However in Gen-LaneNet, 3D are lifted from 2D and thus this consistency is inherent to the pipeline.

The paper also **decoupled** 2D lane line prediction and lifting 2D to 3D. Decoupling 3D structure prediction from 2D LLD: motivated by the fact that encoding of 3D geometry is rather independent from image features. This decoupling makes the module more flexible and scalable. The second stage can be trained with on a synthetic dataset alone. The **deliberate removal of image feature in the second stage** is similar to predicting distance with bbox info only in [Object frontal distance](obj_dist_iccv2019.md), and with skeleton only [MonoLoco](monoloco.md). 

This paper is quite hard to read (like other papers by Baidu, unfortunately), but the underlying ideas are solid. I wished the paper included a clear definition of what is a (real) top view  and a virtual top view.

#### Key ideas
- Architecture
	- The first stage is just lane segmentation
	- The second stage (3D GeoNet) takes in lane segmentation results, perform [spatial transformer network](stn.md) to be a IPM image, and then have the same 3D lane head on top of it. 

#### Technical details
- Lidar LLD dataset essentially only goes up to 48 meters. Hard to obtain GT further away than that. 

#### Notes
- [3D LLD dataset](https://github.com/yuliangguo/3D_Lane_Synthetic_Dataset)
