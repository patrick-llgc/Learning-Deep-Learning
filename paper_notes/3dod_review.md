# [A Survey on 3D Object Detection Methods for Autonomous Driving Applications](http://wrap.warwick.ac.uk/114314/1/WRAP-survey-3D-object-detection-methods-autonomous-driving-applications-Arnold-2019.pdf)

_October 2019_

tl;dr: Summary of 3DOD methods based on monocular images, lidars and sensor fusion methods of the two.

#### Overall impression
The review is updated as of 2018. However there have been a lot of progress of mono 3DOD in 2019. I shall write a review of mono 3DOD soon.

#### Key ideas
> - The main drawback of monocular methods is the lack of depth cues, which limits detection and localization accuracy, especially for far and occluded objects. 

> - Most mono 3DOD methods have shifted towards a learned paradigm for RPN and second stage of of 3D model matching and reprojection to obtain 3D Box. 

#### Technical details
- Mono
	- [mono3D](mono3d.md)
	- 3DVP
	- subCNN
	- [deepMANTA](deep_manta.md)
	- [deep3DBox](deep3dbox.md)
	- 360 panorama
- Lidar
	- projection
		- VeloFCN
		- [Complex YOLO](complex_yolo.md)
		- Towards Safe (variational dropout)
		- BirdNet (lidar point cloud normalization)
	- Volumetric
		- 3DFCN
		- Vote3Deep
	- point net
		- VoxelNet

- Sensor Fusion
	- [MV3D](mv3d.md)
	- [AVOD](avod.md)
	- [Frustum PointNet](frustum_pointnet.md)

#### Notes
- camera and lidar calibration with odometry
