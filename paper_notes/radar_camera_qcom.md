# [Radar and Camera Early Fusion for Vehicle Detection in Advanced Driver Assistance Systems](https://ml4ad.github.io/files/papers/Radar%20and%20Camera%20Early%20Fusion%20for%20Vehicle%20Detection%20in%20Advanced%20Driver%20Assistance%20Systems.pdf)

_December 2019_

tl;dr: Early fusion of radar and camera via range-azimuth map + IPM feature concatenation.

#### Overall impression
This is follow up work on Qualcomm's ICCV 2019 paper on [deep radar detection](radar_fft_qcom.md). The dataset still only contains California highway driving. 

The addition of camera info does not boost the performance of radar a lot (only about 0.05%), and it suffers less if the camera input is set to 0. The camera info did help reducing the lateral error. 

Training the camera branch in advance (similar to [BEV-IPM](bev_od_ipm.md)) and freeze it during joint training yielded the best results. 

#### Key ideas
- Two parallel branches, then concatenating final features from both backbones for detection head. 
	- Radar: range azimuth map
	- Camera: IPM. 
	- Range: 46.8 x 46.8 m
	- The resolution of the feature maps match so they can be concat or added together.
- Empirically, **placing polar to cartesian early in the intermediate feature layers in the radar branch gave the best performance.** --> this is different from the method in [deep radar detection](radar_fft_qcom.md).
- Placing the IPM transformation directly to the camera image works best.
- Camera branch is harder to train than radar branch. The jointly trained network is more robust toward failure of camera branch.
	- visually occluded objects may still return radar signal
	- highway scene is easier for radar as there is limited clutter.
	- Distant objects mostly depend on radar. 

#### Technical details
- Traditional radar literature refers to detection as the task of returning a spatial point, in contrast to the computer vision community where detection usually returns a bbox.
- Synchronization: extract one radar frame per 100 ms, get nearest neighbor camera frame, and use interpolated lidar annotation.
- In lidar, points from objects at the boundary of a frame might have moved significantly. This discontinuityis take care of in lidar processing pipeline.
- Lidar GT is human corrected, non-causal processing and temporal tracking (both forward and backward?) is used. 
- 

#### Notes
- Background
	- traditionally, early fusion allows low-level fusion of the features resulting in better detection accuracy.
	- radar data is typically processed using a CFAR algorithm to convert the raw data into a point cloud which separates the targets of interest from the surrounding clutter.
- Synchronization: why not use lidar as base line? --> Maybe still inherent misalignment between radar and lidar annotation. 
