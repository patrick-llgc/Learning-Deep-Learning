# [FCOS3D: Fully Convolutional One-Stage Monocular 3D Object Detection](https://arxiv.org/abs/2104.10956)

_August 2021_

tl;dr: FCOS baseline of mono3D.

#### Overall impression
The majority of the single stage mono3D methods since [SMOKE](smoke.md) all use [CenterNet](centernet.md) as baseline. This paper switches it to [FCOS](fcos.md) and achieves good results.

Objects are distributed to different feature levels with the consideration if the 2D scales (from reprojected 3D bbox, no 2D annotation is required). 

**The core challenge of mono3D** is how to assign 3D targets to 2D domain with the 2D-3D correspondense and predict them afterwards.

#### Key ideas
- FCOS-like architecture
	- separate heads for classification (+attributes) and centerness. The 3D attributes share the same neck as centerness and each has a small head.
	- 3D attributes
		- Angle: rotation theta (1 ch) and direction classification (2 ch). This is better than centerNet baseline. This is after [SECOND: Sparsely Embedded Convolutional Detection](https://www.mdpi.com/1424-8220/18/10/3337/pdf) <kbd>Sensors 2018</kbd> .
		- Distance: predict log(d) but the loss is in linear space. This gives better performance than loss in log space.
	- Centerness: helps suppress low-quality predictions far away from object centers. 
- Ablation study: baseline only 28.2 NDS all the way to 41.5 NDS.


#### Technical details
- Ambiguity issue: when a point is inside multiple ground truth bboxes in the same feature level. FCOS chooses the bbox with smaller area as the target box. However FCOS3D chooses the box with closer distance. (Prefers objects in the foreground).
- Nuscenes has 1000 scenes.
- Nuscenes does not use IoU but instead uses 2D center distance d on the ground-plane for decoupling detection from object size and orientation. mAP is calculated by averaging AP at different thresholds D = {0.5, 1, 2, 4} m.
- TTA for centerNet or FCOS: averaging score maps by detection heads gives better results than merging bbox at last.

#### Notes
- [Github page](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/fcos3d/README.md)
- How did the network learn velocity with only one frame? This is not reliable at all. Lidar based methods have much lower error for velocity.
- The current model also struggles with big objects and occluded objects. The authors noted that the former may be due to the not sufficiently large receptive field.
