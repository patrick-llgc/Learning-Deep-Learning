# [Consistent Video Depth Estimation](https://arxiv.org/abs/2004.15021)

_July 2020_

tl;dr: Test time training monocular depth network to generate temporally and geometrically consistent depth map.

#### Overall impression
The paper takes pretrained depthNet and performs test-time finetune for each of the test video clip. It is dense (cf. multiview stereo or SfM), globally scale consistent.

**Depth from video** method explicitly enforces geometric constraints. They either use RNN to leverage temporal information ([Dont forget the past](recurrent_depth_estimation.md)) or explicitly using multiview reconstruction (such as [DeepV2D](deepv2d.md), [Neural RGBD](neural_rgbd.md)). 

It is able to handle dynamic object if the movement is consistent (e.g., moving car) and not aligned with epipolar geometry, or it is inconsistent (e.g., waving hands). It works with gentle motion (waving hands) but breaks for extreme object motion. When the movement is aligned with epipolar geometry, it will often cause the infinite depth problem.

#### Key ideas
- Preprocessing
	- Use of SfM pipeline COLMAP for preprocessing mainly focuses on to get a **scale-consistent** **camera pose** pipeline with DepthNet. --> Why not use PoseNet? Maybe it is not accurate enough?
		- Get sparse depth map and camera pose
		- Align depth map scale with deep learning pipeline and scale camera pose translation vector. 
		- If we have up to scale depth prediction pipeline, and accurate camera pose from a localization pipeline, we do not need COLMAP.
	- Mask RCNN to remove dynamic object
- Loss: 
	- image space 2D loss: reprojection with optical flow (from [FlowNet2](flownet2.md)).
	- disparity distance: depth misalignment
- Frame sampling: every 1, 2, 4 images apart, thus O(N) instead of O(N^2).
- The result video has sharp boundaries, thanks to **long range temporal constraints**. 
- Results on finetuning monodepth on KITTI has mixed results. Overall quality is not improved. It improves on 80% of the frames but have large outliers for the other frames. --> This is mainly caused by inaccurate pose estimation. How about using accurate pose from localization?

#### Technical details
- Overlap test: Optical flow consistency check. We do a homography warp first to eliminate rotation. Then perform consistency check, if the consistency mask < 20% of the original image then discard this image pair. For KITTI, the threshold is raised to 50%, due to the large forward motion causing inaccurate flow estimat

#### Notes
- Coupled training with optical flow will lead to better performnance, such as GeoNet.

