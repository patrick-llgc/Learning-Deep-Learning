# [RefinedMPL: Refined Monocular PseudoLiDAR for 3D Object Detection in Autonomous Driving](https://arxiv.org/abs/1911.09712)

_March 2020_

tl;dr: Sparsify pseudo-lidar points for monocular 3d object detection.

#### Overall impression
The paper is based on the work of [Pseudo-lidar](pseudo_lidar.md). The main contribution seems to be the faster processing time, and the performance gain is not huge. 

Both the unsupervised and supervised method identify foreground regions using 2D image, then perform a distance stratified sampler to downsample the point cloud.

#### Key ideas
- Identification of foreground
	- Unsupervised: keypoint detection with laplacian of gaussian (LoG), then keep second nearest neighbors.
	- Supervised: train a 2D object detector and use union of bbox masks.
- Downsampler: downsample uniformly within different distance bins.

#### Technical details
- Distance stratified sampler can maintain the 3d detection performance even with down to 10% samples.
- The performance drop is mainly caused by the distance estimation.

#### Notes
- Questions and notes on how to improve/revise the current work 

