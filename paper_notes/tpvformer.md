# [TPVFormer: Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2302.07817)

_February 2023_

tl;dr: Academic alternative to Tesla's Occupancy Network, by lifting BEVFormer to 3D. 

#### Overall impression
The model uses sparse supervision at training but can predict more consistent and comprehensive volume occupancy for all voxels at inference time. (Denser, but not really dense, as compared to [SurroundOcc](surroundocc.md))

TPV extends the idea of BEV to 3 orthogonal axis, and thus models 3D without suppressing any axes and avoiding cubic complexity.

The architecture is innovative, but the performance suffers from sparse annotation. [SurroundOcc](surroundocc.md) showed that with dense annotation, the performance can be boosted 3x. 

#### Key ideas
- Two steps of attention
	- 1. Image cross-attention (ICA) to get TPV features
		- between TPV grid queries and the corresponding 2D image features to lift 2D to 3D
		- This is generalization of BEV to the other two orthogonal direction
	- 2. Cross-view hybrid attention (CVHA) between TPV features
		- Enhances each TPV features by attending to the other two
		- For each query, the reference points are selected differently
			- For the same TPV, sample the neighborhood in the same TPV
			- For the two perpendicular TPVs, lift to different height in the TPV pillar and then project to the two perpendicular TPV planes. 
		- The results are three enhanced TPV features. 
- From TPV to features
	- Point features: passive sample, project to 3 planes and aggregate them.
	- Voxel features: active broadcast, produce 3 tensors of the same size of HWDC and then aggregate. 
- (Almost) arbitrary resolution adjustment during inference. This is inspired by Nerf and Occupancy Network by Andreas Geiger.

#### Technical details
- Why BEV is good
	- information for outdoor scenes are not isotropically distributed, modern methods collapse the height information and mainly focus on the ground plane (BEV) where info varies the most
	- BEV works surprisingly well as 3D object detection only predict coarse level bbox for cars and pedestrians.
	- BEV reduces computation from O(HWD) to O(HW)
- Why we need 3D occupancy
	- Objects with various 3D structures can be encountered in real scenes and difficult to encode them in a flattened vector
	- Omitting the z-axis has adverse effect on its expressiveness.
- TPV query maps to a 2D grid cell region of sxs m^2 in the corresponding view, and further to a 3D pillar region extending from the view in the perpendicular direction. --> Just like each BEV query. 
- A new interesting track of vision-only lidar segmentation: vision to construct the feature volume, and lidar is only used to query features.
- Segmentation loss is cross entropy + [Lovasz-softmax](https://paperswithcode.com/method/lovasz-softmax). The Lovasz extension is a means by which we can achieve direct optimization of the mean intersection-over-union loss in neural networks. 


#### Notes
- [Code on github](https://github.com/wzzheng/TPVFormer)
- Future directions
	- KPI evaluation of vision-based semantic occupancy prediction needs to be quantified. This will help create a new track for academic and make this work a seminal work in this track.
	- - "Empty" class: different from lidar semantic segmentation where each point queried is valid, sematnic occupancy prediction needs to predict void voxels as well. The paper simply assigns voxels without any points "empty". --> This fails to differentiate between void and uncertain (due to occlusion, e.g.), and perhaps can explain the weird scanning pattern issue in the demo video.
- Q & A 
	- How is a HCAB block constructed? Looks like it has both ICA and CVHA. --> It is indeed per [config](https://github.com/wzzheng/TPVFormer/blob/27627079bbb87ae1b8e0b3acf9a1a8f4cdc81cfe/config/tpv_lidarseg.py#L151).