# [Semi-Local 3D Lane Detection and Uncertainty Estimation](https://arxiv.org/abs/2003.05257)

_March 2020_

tl;dr: Semi-local 3D lane line detection.

#### Overall impression
This paper is from the same authors for [3D LaneNet](3d_lanenet.md), and improved the performance by a large margin.

The parameterization of the lane lines within each patch is by a polar coordinate (r and $\theta$). The advantage of this approach over directly predicting Cartesian offset from the tile center is unclear. In theory these two representation should be equivalent. 

The main advantage of this paper seems to be the introduction of tiles (2D anchors) as compared to 1D anchors in [3D LaneNet](3d_lanenet.md). This helps the model generalize to more complex topologies in a simpler way. In contrast, [3D LaneNet](3d_lanenet.md) deals with merges and splits with regression of two targets per anchor. 

The paper also introduced uncertainty estimation. How to use this uncertainty and how did this help with the lane detection is not clear.

#### Key ideas
- Assumes that one image tile only has one lane going through it, and thus only one embedding. (This may have some limitation, as seen in Fig 5 that some lanes are disconnected.)
- Different tiles intersecting with the same lane has the same embedding.
- The paper trained an **embedding** prediction layer to cluster the tiles for the same lane together. The push-pull loss is on image tiles and have much less computation burden than semantic segmentation. For an image with C lanes, N_c is the number of tiles belonging to that lane: (this loss is directly borrowed from [LaneNet](lanenet.md) which used it on a pixel level)
	- Pull: C * Nc items, pulling Nc points within each lane 
	- Push: C * (C-1) pairs of lanes. 
- During inference, the clustering is done by the mode-seeking algorithm mean-shift (cf [LaserNet](lasernet.md) for more details) to find the center of each cluster and then set threshold to get cluster members. (In [associative embedding](associative_embedding.md))
- As with other algorithms with association, the loss requires association during the network forwarding.

#### Technical details
- The ROI is divided into patches (16x26 patches in 20 m x 80 m area).
- Synthetic data helps with real data training.
- Assuming a fixed homography between image and BEV, doing 2D LLD in 3D, then project to 2D during inference actually achieves quite good performance. 

#### Notes
- Can we directly predict 3D position of lane lines.