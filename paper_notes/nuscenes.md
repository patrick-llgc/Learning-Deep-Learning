# [nuScenes: A multimodal dataset for autonomous driving](https://arxiv.org/abs/1903.11027)

_July 2020_

tl;dr: This is the first large dataset with camera, lidar and radar data.

#### Overall impression
The dataset is quite challenging in many aspects:

- multiple country, multiple city (Boston, Singapore)
- multiple weather condition
- low framerate annotation (2 FPS for camera and lidar, although camera captured at 15 FPS and lidar 20 FPS). This makes tracking harder.

#### Key ideas
- 1000 scenes, each 20 seconds long.
- Revised mAP (different from KITTI) for 3D OD. 
	- We use the Average Precision (AP) metric [32, 26], but define a match by thresholding the 2D center distance d on the ground plane instead of intersection over union (IOU). This is done in order to decouple detection from object size and orientation but also because objects with small footprints, like pedestrians and bikes, if detected with a small translation error, give 0 IOU (Figure 7). This makes it hard to compare the performance of vision-only methods which tend to have large localization errors [69].
	- Same convention is used in [Argoverse](argoverse.md).

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

