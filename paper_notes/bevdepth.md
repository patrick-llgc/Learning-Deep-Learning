# [BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection](https://arxiv.org/abs/2206.10092)

_July 2022_

tl;dr: LSS with explicit depth supervision and Efficient Voxel Pooling.

#### Overall impression
In LSS, depth estimation is implicitly learnt without camera info. The accuracy of depth estimation is surprisingly inadequate (pred-gt curve shows very poor correlation). Replacing the depth with ground truth depth will lead to huge improvement, indicating that the quality of intermediate depth is the key to improving multi-view 3D object detection.

#### Key ideas
- Explicit depth supervision: The depth is supervised by projected point clouds, with a *min poling* and a *one-hot* encoding.
- Depth correction is introduced to account for the error in extrinsics calibration (and online extrinsic disturbance). Instead of finding the exactly correct calibration, BEVDepth proposes to **enlarge the receptive field to compensate for inaccurate lidar-camera alignment**. In this way, the mis-aligned depth GT is able to attend to the feature at right position. --> This is a good point. May be very useful for early fusion.
- Camera aware depth prediction --> This is an alternative to [CamConv](cam_conv.md)
	- Use MLP to scale up the dimension of intrinsics to higher dimension (e.g., 128d)
	- Do the same for extrinsics (R and t), and concatenate with intrinsics feature. 
	- Then the camera calib feature is used to weight camera features with SENet (squeeze and excitation) module.

#### Technical details
- Efficient Voxel Pooling --> See also in [BEVFusion](bevfusion.md) and [M2BEV](m2bev.md).
	- Assigning each grid in BEV to a thread, and each thread adds the associated point features in parallel. 
	- This speeds up the Voxel Pooling in [LSS](lift_splat_shoot.md). The voxel pooling operation alone is speed up x100. Overall pipeline speed by x3.
	- The **cumsum trick** used by [LSS](lift_splat_shoot.md) cannot run in parallel.
- Multiframe
	- Ego motion compensation of the pseudo-lidar point cloud of the previous frame, then aggregated with the current frame.
	- Time window is only 2, same to [BEVDet4d](bevdet4d.md).

#### Notes
