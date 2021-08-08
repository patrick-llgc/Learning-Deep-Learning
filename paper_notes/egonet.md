# [EgoNet: Exploring Intermediate Representation for Monocular Vehicle Pose Estimation](https://arxiv.org/abs/2011.08464)

_August 2021_

tl;dr: Patch-based refinement module to predict vehicle orientation based on wireframes.

#### Overall impression
EgoNet extracts heatmaps of object parts from local object appearances, which are mapped to the screen coordinates and then further lifted to 3D object pose. 

The wireframe (or sparse 3D point cloud representating an interpolated cuboid) idea is similar to that of [RTM3D](rtm3d.md) and [KM3D-Net](km3d_net.md) and [FQNet](fqnet.md).

#### Key ideas
- Architecture
	- FCN to extract 2D heatmaps of k keypoints. This is supervised by 2D heatmap loss.
	- FCN to convert heatmaps to k patch coordinates.
	- Affine transformation (scaling and translation) from patch coordinates to image coordinates
	- Lifter MLP to lift to k-1 camera coordinates. As the translation is irrelevant to pose estimation, 3D coordinates are normalized relative to the centroid.
	- Postprocessing: lifting 3D points to 3D orientation. --> This step is NOT detailed in the paper.
- Cross-ratio loss function: invariant to projection, and can be used based on unannotated images.
	- cross-ratio ((v3-v1)(v4-v2))/((v3-v2)(v4-v1))
	- used unannotated apollo3D patches (still need 2d bbox though) to help with model generalization
- Loss
	- Heatmap loss
	- 2D loss (in image plane)
	- 3D loss (in camera coordinate system)
	- Cross-ratio (CR) loss

#### Technical details
- EgoNet can be used as a module to improve existing single-stage 3D object detectors. --> It can be applied to both camera or lidar input as well.
- The estimation from 3D points to Euler is implemented by SVD ([code](https://github.com/Nicholasli1995/EgoNet/blob/master/libs/common/transformation.py#L99)).
- Even with only 9 points, it is possible to do self-supervised learning.
- Based on a perfect detection, AOS of EgoNet can go up to 99%, significantly higher than

#### Notes
- [Code on github](https://github.com/Nicholasli1995/EgoNet)
- [Youtube video](https://www.youtube.com/watch?v=isKo0F3MU68)
