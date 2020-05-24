# [PseudoLidarV3: End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection](https://arxiv.org/abs/2004.03080)

_May 2020_

tl;dr: End to end finetuning of depth prediction network with 3d object detector.

#### Overall impression
The paper proposed end to end method to train depth net and 3d object detector in [pseudo-lidar](pseudo_lidar.md). It is different from CMU's [end to end pseudo lidar approach](pseudo_lidar_e2e.md), and proposed two methods to truly make the depth lifting differentiable.

The paper praised [pseudo-lidar](pseudo_lidar.md) for its plug-and-play modularity but it also pointed out that PL suffers from the sub-optimal solution from proxy losses. Only 10% of point clouds belong to foreground (cf [ForeSee](foresee_mono3dod.md), and PLv3 targets to finetune these points.

The sparsification is based on r, azimuth and elevation angle. A better approach may be [Refined MPL](refined_mpl.md).

The paper explored two representative lidar object detector, one one-stage method based on voxelization (PIXOR) and the other two-staged method based on point cloud directly ([PointRCNN](point_rcnn.md))

> Although it will probably always be beneficial to include active sensors like LIDARs in addition to passive cameras, it seems possible that the benefit may soon be too small to justify large expense.

#### Key ideas
- **CoR (Change of representation)**:
	- Quantization: instead of hard binning, use gaussian blur and summation. In other words, a sample does not only affect the nearest bin it falls into, it also impact neighboring bins weighted by a Gaussian kernel (referred to as radial basis function RBF). This **differentiable point cloud renderer** is very similar to the differentiable rendering used in [SynSin](SynSin: End-to-end View Synthesis from a Single Image) <kbd>CVPR 2020</kbd>.
	- SubSampling: as long as we properly document which points are subsampled, the gradient can be propagated back to those points.
- The loss gradient map shows that the network is indeed finetuning (soft *attending to*) minority points in the foreground. 
- Finetuning each module with the other module in place does not leads to better performance for pseudo-lidar. Instead, having the end-to-end finetuning enabled by CoR is critical, especially for PIXOR. 

#### Technical details
- Due to the sparsity in subsampled point cloud, training with depth loss is required. 
- PIXOR still lacks behind PointRCNN for performance.

#### Notes
- Questions and notes on how to improve/revise the current work  

