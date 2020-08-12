# [AVP-SLAM Late Fusion: Mapping and Localization using Semantic Road Marking with Centimeter-level Accuracy in Indoor Parking Lots](https://ieeexplore.ieee.org/abstract/document/8917529)

_August 2020_

tl;dr: SLAM in parking lot with road markings.

#### Overall impression
This work is almost exactly the same similar to [AVP-SLAM](avp_slam.md). Both  use road marking for mapping and localization in parking lot. Both processes 4 synchronized (or almost synchronized) fisheye images into a stitched surround view. 

However it is different from [AVP-SLAM](avp_slam.md) in that it proposes to do semantic segmentation in the fisheye images first, perform IPM, and then do motion compensation to stitch the semantic point cloud. 


#### Key ideas
- Accumulating all the points in the semantic map is problematic, first due to map size, second due to errors in single image results. 
- Single frame point cloud inevitably has some errors in semantic segmentation, and a semantic point cloud fusion step is used to fuse the single frame semantic point cloud into a coherent map. --> This is not explicitly mentioned in [AVP-SLAM](avp_slam.md) but most likely is done as well.
- Loop closure with ORB-features and also semantic point cloud features. 
- Global initialization is done with tracking multiple candidates until converges. --> Maybe we can also use GNSS or parking lot entrance for initialization.

#### Technical details
- To keep point cloud registration algorithm runs at constant time, random discarding of semantic points is needed. --> This is perhaps not needed when only keypoints are extracted instead of a dense semantic segmentation. 
- 7 semantic classes are labeled: white line, zebra crosswalk, arrow, speed bump, yellow line, others. 
- Evaluation of localization: 
	- **repeatability is more important than absolute accuracy**. (Or, we care more about localization error than mapping error, per [AVP-SLAM](avp_slam.md).) 
	- Localization trajectories are aligned with reference trajectory (during mapping) and calculate trajectory error.5

#### Notes
- Image stitching or semantic point cloud stitching? Image stitching perhaps will be more useful as it can formulate the data bases if more types of semantic labels are needed. It also decouples the dependency of onboard model performance with offline map generation. 

