# [3DSSD: Point-based 3D Single Stage Object Detector](https://arxiv.org/abs/2002.10187)

_January 2021_

tl;dr: Single-stage, anchor-free point-based lidar object detector.

#### Overall impression
This study brings the point-based lidar object detectors ([PointRCNN](point_rcnn.md)) to the realtime realm. Almost all previous real-time lidar object detectors are voxel based, such as [PIXOR](pixor.md), [PointPollars](point_pillars.md) and [VoxelNet](voxelnet.md).

Existing point based lidar object detector have largely four stages

- Set abstraction (SA): downsampling to get representative points
- Feature propagation (FP): upsampling
- RPN on each point
- Refinement stage

The feature propagation and refinement module takes more than half of the time and thus need to be eliminated. The largest contribution of 3DSSD is the use of feature based furthest point sampling to eliminate the need for feature propagation. --> This reminds me of [feature-metric loss](feature_metric.md) <kbd>ECCV 2020</kbd> in monocular depth estimation, and [BEV feat sticthing](bev_feat_stitching.md).

The paper introduces quite a few acronyms which makes the paper a bit hard to read.

#### Key ideas
- Set abstraction (SA) sampling: it takes the form of **furthest point sampling (FPS)** based on Euclidean distance, or D-FPS. 
	- The downsampled representative points by D-FPS discards too many points on the FG objects. Thus FP layer has to be used to recall the discarded points. 
	- Feature-FPS takes in the semantic feature distance to solve this issue. 
- Fusion sampling (FS): A combined feature-FPS and distance-FPS yields the best performance. 
- Candidate generation (CG): shift the points selected by F-FPS to the center, by a dedicated supervision loss. The shifted **representative points** are called **candidate points**.
- Anchor-free head: for each candidate point, predict distance offset to the center of the corresponding instance. Less cumbersome than anchor-based method
- Losses:
	- classification loss: cls conf and centerness score with CE loss
	- regression loss: distance, orientation, size, corner loss
	- shift loss to shift representative point towards the center.

#### Technical details
- Centerness assignment: inspired by [FCOS](fcos.md), every candidate points also regresses one centerness score to describe how far away it is from the center of the predicted object. 
	- Although not explicitly described, this centerness score is most likely used together with cls conf in postprocessing to filter out FPs.
	- The original points before shifting is not suitable to regress this centerness score as almost all points are on the surface of the object and are all quite far away from the center. Without shifting, predicting centerness actually hurts the performance.


#### Notes
- Questions and notes on how to improve/revise the current work  

