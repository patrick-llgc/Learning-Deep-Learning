# [VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion](https://arxiv.org/abs/2302.12251)

_March 2023_

tl;dr: class agnostic query proposal, plus class specific semantic segmentation. 

#### Overall impression
Key intuition: Visual features on 2D images corresponds only to the visible scene structures rather than the occluded or empty space. It uses the bottom-up depth estimation as the scaffold for 3D scene understanding.

The SSC (semantic scene completion) has to address two issues simultaneously: **scene reconstruction for visible areas and scene hallucination for occluded regions**. 

> Real world sensing (cf perception) in 3D is inherently sparse and incomplete. For holistic semantic understanding, it is insufficient to solely parse the sparse measurements while ignoring the unobserved scene structures. 

The paper performs depth estimation with monodepth methods first, lift to pseudo-lidar point cloud, then voxelize them into initial query proposals. These sparse queries, coupled with learned masks, use self-attention to densify the sparse prediction.

**Why occupancy?** --> Occupancy for each cell instead of assigning a fixed size bounding box to an object, could help identify an irregularly-shaped object with an overhanging obstacle. 

#### Key ideas
- Two stages
	- class agnostic query proposal, with monodepth
	- class specific semantic segmentation --> This is essentially label propagation.
		- Update sparse non-empty voxel queries (Q_p) with image features via cross-attention,
		- Update all voxels (Q_p and mask token m) via self-attention. This is the **densification** process. --> cf supervision via dense labeling in [OpenOccupancy](openoccupancy.md), and the semi-dense prediction results from [TPVFormer](tpvformer.md) learned with sparse supervision.
- Temporal information does not help with geometric occupancy prediction (IoU), but improves semantic segmentation (mIoU) quite a lot. 
- This work heavily depends on accurate depth prediction, and could benefit from the advancement of depth estimation.
- Short range vs long range
	- understanding of a short-range area is more ***crucial*** since it leaves less time for autonomous vehicles to improve. 
	- Understanding of a ***provisional*** long-range area could be enhanced as ego car get closer to collect more observation.


#### Technical details
- Mask tokens are basically learned embeddings, and they are used to pad the sparse queries in spatial domain to form a uniformly shaped dense tensor, before feeding into self-attention. In other words, |Q_p| + |m| = |Q|.
- 256x256x32, with 0.2m resolution. 20 classes (19 semantic + 1 empty). --> Same as [OpenOccupancy](openoccupancy.md).
- KPI evaluation: IoU for geometric completion, and mIoU for semantic segmentation.

#### Notes
- Work from nVidia and Sanja Fidler's team seems to like the "bottom-up" approach to use depth estimation for 3D scene understanding. Similar work include [Lift-Splat-Shoot](lift_splat_shoot.md).