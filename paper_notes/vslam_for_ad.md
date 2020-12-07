# [Visual SLAM for Automated Driving: Exploring the Applications of Deep Learning](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/Milz_Visual_SLAM_for_CVPR_2018_paper.pdf)

_December 2020_

tl;dr: An overview of deep learning application in SLAM for autonomous driving.

#### Overall impression
The paper provides a good overview of VSLAM applications in autonomous driving.

#### Key ideas
- Three main scenarios for autonomous driving: **highway, parking lot and city**
	- Parking: needs an accurate environment map in the near vicinity of the car while driving at low speed.
	- Highway: 20 FPS or more. 
		- Orb-SLAM is doing a great job already in highway scenes.
	- City: Many dynamic objects that needs to be detected actively or passively during 3D reconstruction.
		- Orb-SLAM performs poorly. The ability to reconstruct static points with stability against lots of dynamic objects within the scene is key.
- Two approaches to autonomous driving
	- Mediated perception approach
	- End-to-end approach. Perception can be used as auxiliary supervision.
- Two main types of HD map
	- Dense semantic point cloud maps
		- **TomTom, Google**
		- Mapped with lidar/camera
		- Provides strong prior to semantic segmentation
	- Semantic landmarks maps
		- **MobileEye, HERE**
		- Mapped with camera
- Private small scale map 
	- Small scale mapping capability is necessary
	- Privacy, Coverage and dynamic
- Fundamental pipelines of SLAM
	- Tracking (Visual Odometry, frontend)
	- Mapping: sparse (feature based method) or dense (direct method)
	- Global optimization and loop closure
	- Re-localization
- History of SLAM
	- Feature based SLAM
		- MonoSLAM: EKF-tracking
		- PTAM: bundle adjustment
		- Orb-SLAM: loop closure + global pose optimization
	- Direct SLAM
		- DTAM: photometric error
		- LSD-SLAM: loop closure + global pose optimization
		- DSO: geometric error, lens distortion, exposure time calib
- Deep learning opportunities in SLAM
	- depth estimation
	- optical flow
	- feature correspondence
	- bundle adjustment
	- semantic segmentation
	- camera pose estimation


#### Technical details
- Stereo SLAM are acceptable for autonomous driving applications, but monocular results are weak and unacceptable.
- Rolling shutter has to be accounted for on highways for accurate SLAM
- There is still no mature solutions for how to do self-repairing map, and map on the vehicle.
- We need motion segmentation for general obstacle detection.

#### Notes
- Do we really need 20 FPS for parking lot and city?

