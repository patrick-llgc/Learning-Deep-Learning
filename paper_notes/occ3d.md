# [Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving](https://arxiv.org/abs/2304.14365)

_May 2023_

tl;dr: Creation of two semantic occupancy prediction datasets, Occ3D-Waymo and Occ3D-nuScenes.

#### Overall impression
Occ3D also proposed a pipeline to generate dense occupancy labels, which includes point cloud aggregation, point labeling, and occlusion handling. The visibility and occlusion reasoning of the label is the main contribution of the paper. 

It does not have the densification process in [SurroundOcc](surroundocc.md) and [OpenOccupancy](openoccupancy.md), which focused on NuScenes dataset. The authors claim that the label is already quite dense even without densification for Waymo dataset, and Poisson Recon leads to inaccurate annotation.

The paper also proposed a neural network architecture Coarse-to-Fine Occupancy (CTF-Occ). This is largely the same as Cascade Occupancy Net (OCNet) by [OpenOccupancy](openoccupancy.md) and the coarse to fine architecture of [SurroundOcc](surroundocc.md). It proposes two tricks: incremental token selection to reduce computation burden, and an implicit decoder to output the semantic label of any given point, similar to the idea of [Occupancy Networks](occupancy_networks.md). 


#### Key ideas
- Each voxel is described by `occupancy` (occupied or free) and `semantics` (class, or unknown). GO (general obstacles) can be described as (occupied, unknown), and a free space voxel can be described as (free, None)
- SCC/OGM vs SOP: Conventional OGM requires measurements from range sensors like LiDARs and RADARs, and also makes the assumption that the scene is static over time. Semantic Occupancy prediction can be used in vision-only systems in dynamic scenes. 
- Step 1: Point cloud aggregation
	- By leveraging ego poses and object tracks, point cloud aggregation and dynamic object transformation enhance the static scene’s density and recover the detailed geometry of dynamic objects.
- Step 2: Visibility and occlusion reasoning
	- Ray-casting-based methods to estimate both LiDAR and camera visibility, as visibility masks are crucial for evaluating the 3D occupancy prediction task.
	- Lidar visibility describes the completeness in the GT. The GT of Some voxels are not observable, even after multiframe data aggregation.
	- Camera visibility focuses on the possibility of detection of onboard sensors. 
	- Eval is only performed on the “observed” voxels in both the LiDAR and camera views.
	- Note that the LiDAR visibility mask and camera visibility mask may differ due to two reasons: (1) LiDAR and cameras have different installation positions; (2) LiDAR visibility is consistent across the whole sequence, while the camera visibility differs at each timestamp.
- CTF-Occ
	- Incremental token selection. Essentially it selects a sparse subset of occupied voxels as token (queries) to reduce computational and memory cost. A binary occupancy classifier is trained, then the top-K most uncertain voxel tokens are selected. 
	- Spatial cross attention is roughly the same as [BEVFormer](bevformer.md).
	- **The implicit decoder** can offer output at arbitrary resolution. It is implemented as an MLP that outputs a semantic label by taking two inputs: a voxel feature vector extracted by the voxel encoder and a 3D coordinate inside the voxel.
	- Loss function has two parts, binary occupancy mask and semantic prediction loss (CE).

#### Technical details
- nuScenes is captured at 10 Hz, but annotated at 2 Hz. To get dense label for dynamic objects, the annotated object box sequence are interpolated across time to auto-label the unannotated frames before performing dynamic point aggregation. 
- Occ3D-Waymo range [-40m, 40m] X and Y axis, and [-5m, 7.8m] along Z axis. Occ3D-nuScenes range [-40m, 40m] X and Y axis, and [-1m, 5.4m] along Z axis.

#### Notes
- [Github](https://github.com/Tsinghua-MARS-Lab/Occ3D)
