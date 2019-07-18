# [2D Car Detection in Radar Data with PointNets](https://arxiv.org/abs/1904.08414)

_July 2019_

tl;dr: Use F-pointnet for car detection with sparse 4D radar data (x, y, $\tilde {v}_r$, $\sigma$).

#### Overall impression
From U of Ulm. Only one target per per car, in a controlled environment. A high precision GPS is used to create the dataset GT.

This is an extension to the [radar point cloud segmentation](radar_point_semantic_seg.md).

#### Key ideas
- Three steps:
	- Patch Proposal around each point --> this proposal is quite like [point rcnn](point_rcnn.md).
	- Classify patch
	- Segment patch (point cloud segmentation)
	- Bbox estimation

#### Technical details
- Radar data often contain reflections of object parts not directly visible, like the wheel house (fender) on the opposite side.
- No accumulation of data across frames like the radar point segmentation work.


