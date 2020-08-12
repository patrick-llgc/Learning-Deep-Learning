# [AVP-SLAM: Semantic Visual Mapping and Localization for Autonomous Vehicles in the Parking Lot](https://arxiv.org/abs/2007.01813)

_August 2020_

tl;dr: Semantic feature on the ground for mapping and localization in parking lots. Similar idea to [Road SLAM](road_slam.md).

#### Overall impression
This paper is on localization and mapping in parking lot, but the same principles apply to urban environment, which are also narrow, crowded and GPS-denied. 

The semantic features are robust to perspective or illumination changes, long-term stable (as compared to traditional features such as ORB in [ORB SLAM](orb_slam.md)). This helps achieve cm-level accuracy required for AD.

This paper is extremely similar to a similar one from SJTU, [AVP-SLAM-late-fusion](avp_slam_late_fusion.md). [AVP SLAM](avp_slam.md) requries synchronized image feeds, and [AVP-SLAM-late-fusion](avp_slam_late_fusion.md) explicitly handles unsynchronized images through late fusion of semantic point clouds. 

The paper is very well written and easy to follow.

#### Key ideas
- Four fisheye images from AVP cameras are stitched into one BEV image.
- The input to segmentation map is a stitched IPM image. 
- Mapping:
	- Semantic segmentation results are lifted into 3D and aggregated into a local map. A local map for a 30 meter window is maintained.
	- Loop closure to address long time drift of odometry sensors. Once a loop is detected, a global pose optimization is performed. 
- Localization:
	- When vehicles are visiting the same parking lot again, semantic features are lifted into 3D space and are compared with the map with ICP. 
	- **Good initialization** is needed for ICP (as ICP matched points?): we can use parking lot entrance or GPS for initialization.
	- EKF fuses visual localization results with odometry which guarantees to have a smooth output. Odometry used for prediction, and visual localization is used for updating.
- Evaluation: **localization error is more important than mapping error**. Even an inaccurate map can be used to guide car into parking into the right parking spot as long as the vehicle can precisely localize in this map. 
	- Absolute mapping error meter level. RMSE 4 m in 300 meter length. 
	- Localization error is cm level. Max error 5 cm. 

#### Technical details
- AVP: automatic valley parking
- 6 semantic classes are labeled: lanes, parking lines, guide signs, speed bumps, free space, obstacles and walls. 
- Evaluation with RTK-GPS in open outdoor parking lot. 
- The evaluation is done on datasets collected 1hr, 3hrs, 1day, 1week and 1month apart to demonstrate the robustness of the approach.

#### Notes


