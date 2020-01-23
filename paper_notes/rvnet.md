# [RVNet: Deep Sensor Fusion of Monocular Camera and Radar for Image-based Obstacle Detection in Challenging Environments](https://www.researchgate.net/profile/Vijay_John3/publication/335833918_RVNet_Deep_Sensor_Fusion_of_Monocular_Camera_and_Radar_for_Image-based_Obstacle_Detection_in_Challenging_Environments/links/5d7f164e92851c87c38b09f1/RVNet-Deep-Sensor-Fusion-of-Monocular-Camera-and-Radar-for-Image-based-Obstacle-Detection-in-Challenging-Environments.pdf)

_January 2020_

tl;dr: Fuse radar to camera with sparse pseudo-image as input and two output branches for small and large object detection.

#### Overall impression
This paper uses similar method to convert radar pins into pseudo-image as in [distant object detection](distant_object_radar.md). It is called "sparse radar image".

**Critics:** The "dense radar image" does not make sense to me as it warps a 169-dim feature into a 13x13 image and apply 2D conv. There is no guarantee of the order in the 169-d feature and thus 2D conv seems a bit random.

#### Key ideas
- Input: Image (416x416x3), Radar (416x416x3) (depth, lateral velocity, longitudinal velocity). Both velocities are compensated by ego velocity. Alternatively, radar can be warped into a 13x13 pseudo image directly, which does not make sense. (see above).
- Architecture
	- Image features extracted by TinyYolov3
	- Radar features extracted by a VGG like structure.
	- Feature maps fused at 13x13 (downsize x8) level and has big object detection branch
	- Fused feature map upsampled by 2 to 26x26 and has small object detection
- Fusing radar leads to better object detection. However it did not boost individual class detection. Radar signal is not good at identifying classes in a multi-class classification setup.

#### Technical details
- RVNet uses fixed radar number, 169, which is a bit puzzling. The number of points in the radar data per frame is not fixed. Using [this class](https://github.com/nutonomy/nuscenes-devkit/blob/db2df52aca557a9f81f25235a0618f4f98faa162/python-sdk/nuscenes/utils/data_classes.py#L259) to load nuscenes radar pcd data, and the point cloud is in format of 18 (attr. num) x `num_points`, and `num_points` varies (63, 47, 48, etc).
- Sparse radar image is better than dense radar image across the board (as expected). 
- The one with two branches is better than one detection branch

#### Notes
- Questions and notes on how to improve/revise the current work  

