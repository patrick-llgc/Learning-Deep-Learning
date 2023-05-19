# [MonoScene: Monocular 3D Semantic Scene Completion](https://arxiv.org/abs/2112.00726)

_March 2023_

tl;dr: Single-cam semantic scene completion (SSC) with feature line-of-sight projection.

#### Overall impression
The paper is one of the pioneering papers to expand scene completion from indoor to outdoor. Before this paper most of semantic scene completion work focuses on indoor scenes, such as [CoReNet](corenet.md). It is quite similar to (seems to be a simplified version of) voxel-pooling in [Lift-Splat-Shoot](lift_splat_shoot.md) as well.

The paper proposes a lot of bells and whistles, but seems that only one design choice is critical, **FLoSP (feature LoS projection)**. This is an improved version of the "ray-traced skip connection" proposed by [CoReNet](corenet.md). All other ideas such as Context Relation Prior, and other novel losses (scene-class affinity, local frustum proportion) are quite esoteric and not that useful.

The paper does not have a lot of novelty in my opinion. However, the engineering project is quite well maintained on github, with a demo on HuggingFace.

Therefore, a popular and pioneering work = 1) good selection of topic and 2) open-sourced with engineering quality.

#### Key ideas
- Feature Line of Sight Projection (FLoSP)
	- Very similar to [CoReNet](corenet.md) and [OFT](oft.md), and a simplified version of voxel-pooling in [Lift-Splat-Shoot](lift_splat_shoot.md)
	- The "ray-traced skip connection" in [CoReNet](corenet.md) projects each 2D feature map to a given 3D map, acting as 2D-3D skip connection
	- FLoPS lifts multiscale 2D features to a single 3D feature map
- 3D Context Relation Prior --> Idea interesting, but not that useful.
	- predicts a HWDxHWD affinity matrix of all pixels
	- To boost memory efficiency, simplifies the dense voxel-voxel relationship to Supervoxel-voxel relationship. This reduces memory by a factor of s^3 to HWDxHWD/s^3, where s is the stride.
	- Loss for the affinity matrix is cross-entropy.
	- Similar to the idea of self-attention for completion in [VoxFormer](voxformer.md).
- Other tricks
	- Scene-class affinity loss optimizes the intra-class and inter-class scene-wise metrics.
		- Directly optimize the scene and class-wise metrics (Precision and Recall)
	- Frustum proportion loss aligns the classes distribution in local frustums. 
		- This is to improve the bleeding effect. Occluded voxels tend to be predicted as part of the object that shadows them.

#### Technical details
- SemanticKITTI grid 0.2 m, 256x256x32, 50mx50m, to the front. 
- KPI is also IoU and mIoU. 
- The paper is the first model to use RGB images for SSC. It "invents" baselines by predicting monodepths to adapt other modalities with RGB input. 

#### Notes
- [Github](https://github.com/astra-vision/MonoScene)
