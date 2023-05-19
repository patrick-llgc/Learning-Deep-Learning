# [SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving](https://arxiv.org/abs/2303.09551)

_May 2023_

tl;dr: Dense annotation generation for 3D Occupancy Prediction.

#### Overall impression
Occupancy grid can describe real-world objects of arbitrary shapes and infinite classes. SurroundOcc proposed a pipeline to generate dense occupancy GT without expansive occupancy annotation by human labelers. The paper also demonstrate very clearly that with denser label, previous method ([TPVFormer](tpvformer.md))'s performance can be significantly boosted, almost by 3x. This is the largest contribution of this paper. 

The paper is from the same group of [SurroundDepth](surrounddepth.md), which performs bottom-up depth estimation from the standpoint of the **source**. In comparison, [SurroundOcc](surroundocc.md) performs occupancy prediction from the standpoint of the **target**. This relationship is quite similar to that between [Lift-Splat-Shoot](lift_splat_shoot.md) and [BEVFormer](bevformer.md).

#### Key ideas
- Architecture largely recycles [BEVformer](bevformer.md) with deformable attention.
- Automatic annotation of dense occupancy labels
	- To get dense predictions, we need dense occupancy labels.
	- By stitching multi-0frame lidar points of dynamic objects and static scenes separately. P = [P_s, P_o]
	- Densification with Poisson Reconstruction
	- P is already denser than single frame measurement, but still has many holes. Reconstruct P to a triangular mesh M via **Poisson Surface Reconstruction**. 
	- Poisson Surface Recon: input is point cloud with normal vectors, output is triangular mesh. 
	- M = {V, E}, where vertices V is evenly distributed, then we can convert this to dense voxels Vd. Compared with Vs the direct voxelization from P, Vd is much denser. 
	- For each voxel in Vd, then use **NN (nearest neighbor)** to assign the semantic labels in Vs.
- cross-entropy loss and scene-class affinity loss as proposed in [MonoScene](monoscene.md). High resoltion prediciton is more important, a decayed loss weight 1/2^j for j-th level supervision.

#### Technical details
- Most 3D scene reconstruction (Atlas, NeuralRecon, TransformerFusion) methods are focused on indoor scenes. 
- CD is accuracy and complementary accuracy average, and thus is permutation invariant. CD and F1 score both consider precision and recall.
- Range is [-50m, 50m] x [-50m, 50m] x [-5m, 3m]. 200x200x16 with 0.5m voxel size.
- [TSDF fusion](https://github.com/andyzeng/tsdf-fusion-python) fuses depth measurements into coherent point clouds.

#### Notes
- [Code on Github](https://github.com/weiyithu/SurroundOcc)
