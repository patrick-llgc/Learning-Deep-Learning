# [Multi-Level Fusion based 3D Object Detection from Monocular Images](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Multi-Level_Fusion_Based_CVPR_2018_paper.pdf) (MLF)

_Mar 2019_

tl;dr: Estimate depth map from monocular RGB and use the depth info for 3D object detection.

#### Overall impression
This paper seems to inspire a more influential paper, [pseudo-lidar](pseudo_lidar.md). Especially, Figure 3 basically has the idea of projecting depth map to point cloud, but it was only used for visualization of the detection results. From Fig. 3 it is quite natural to think about object detection with this pseudo-point cloud.

#### Key ideas
- The paper directly concats the depth estimation with RGB to form RGBD 4-ch image and perform object detection on it (input fusion).
- The second branch of a point cloud (XYZ map) contributes to feature fusion.
- The paper also investigated using stereo as input, and the precision is much better than mono. This is the same conclusion as the pseudo-lidar paper.

#### Technical details
- [Summary of technical details]

#### Notes
- [Questions and notes on how to improve/revise the current work ]

