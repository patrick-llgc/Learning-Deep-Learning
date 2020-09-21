# [ROI-10D: Monocular Lifting of 2D Detection to 6D Pose and Metric Shape](https://arxiv.org/abs/1812.02781)

_August 2019_

tl;dr: Concat depth map and coord map to RGB features + 2DOD + car shape reconstruction (6d latent space) for mono 3DOD.

#### Overall impression
Like [MLF](mlf.md), this paper only concat D to RGB, making the performance sub-optimal.

Surprisingly the shape can be approximated pretty well even with 1D (scaling factor). 6D is chosen for this paper to include more details. However this work still concats depth to RGB features instead of lifting RGB into point cloud. This is clearly inferior to other SOTA methods such as [pseudo-lidar](pseudo_lidar.md) and [pseudo-lidar++](pseudo_lidar++.md).

The idea of compressing shapes is also found in [Mask Encoding Instance Segmentation](meinst.md).

The paper articulated that depth has to be reasoned globally. 

#### Key ideas
- 6 DoF pose + 3 DoF size + 1 DoF shape (but more like 6 dim) = 10D ROI. Lifting R^4 -> R^(8x3) from 2D bbox to 8 corners. 
- **Estimating absolute translation is ill-posed due to scale and reprojection ambiguity**. In contrast, Global estimation of depth can be done due to geometric constraint as supervision.
- Use superdepth as OTF depth estimation module. 
- **Architecture: coord-map + depth map + feature map, RoI aligned to generate 3D detections. --> this is inferior to lifting everything to 3D space!**
- When estimating the pose from monocular data only, little deviations in pixel space can induce big errors in 3D. Corner loss is used to regularize overall learning.
- Shape of the car can be described by a truncated signed distance fields (TSDF) of size 128x128x256 (voxelized model). The car undergo an 3D autoencoder to encode the shape into 6 d latent space. The encoded shape vector is on a hypersphere.
- Synthetic 3D data augmentation

#### Technical details
- Latent space of car shapes 
	- [3D RCNN](3d_rcnn.md) 10-dim
	- [RoI10D](roi10d.md) 6-dim
	- [monoDR](monodr.md) 8-dim
- The 3D extents as deviation from the mean extents over the whole dataset (zero-meaned and normalized). 
$$B = q [\pm w/2, \pm h/2, \pm l/2]^T q^{-1} + K^{-1} [xz, yz, z]^T$$
- Ego-centric and allo-centric pose: ego-centric pose can vary according to observation angle, but allo-centric pose always keeps the same for the same type of car (in the local coordinate system of the car). Therefore q is regressed in allocentric coordinate.
- It is much easier to recognize cars if the yaw is 45 degrees or 135 degrees. --> radar looks the same. Maybe corner radar is easier after all.
- Adaptive multi-task weighting by Alex Kendall did not help, but better formulated corner loss helps. 

#### Notes
- The paper mentioned that 3D and 2D NMS matters a lot. --> Can we use center net to do mono 3DOD?
- How much performance boost is from coord conv is unclear. See [coordconv](coord_conv.md) for an interesting technique.
