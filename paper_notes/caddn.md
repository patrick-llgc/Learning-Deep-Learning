# [CaDDN: Categorical Depth Distribution Network for Monocular 3D Object Detection](https://arxiv.org/abs/2103.01100)

_March 2021_

tl;dr: Lift perspective image to BEV for mono3D, with direct depth supervison. Similar to [OFT](oft.md).

#### Overall impression
CaDDN focuses on accurate prediction of depth to improve mono3D performance. The depth prediction method is based on improved version of ordinal regression.

The idea of projecting perspective image to BEV representation and then perform object detection is quite similar to that of [OFT](oft.md) and [Lift Splat Shoot](lift_splat_shoot.md). These implicit method transforms feature map to BEV space and suffers from feature smearing. [CaDDN](caddn.md) leverages probabilistic depth estimation via categorical depth distribution.

Previous depth prediction is separated from 3D detection during training, preventing depth map estimates from being optimized for detection task. In other words, not all pixels are created equal. Accurate depth should be prioritized for pixels belonging to objects of interest, and is less important of background pixels.  [CaDDN](caddn.md) focuses on generating an interpretable depth representation for mono3D. --> cf [ForeSeE](foresee_mono3dod.md)

#### Key ideas
- Depth distribution supervision
	- Dense depth map from depth completion by lidar ([IP-basic](https://github.com/kujason/ip_basic) <kbd>CRV 2018</kbd>)
	- This depth supervision is critical in improving the detection performance. 
	- Sharp depth prediction via one-hot encoding. Ordinal regression. SID (space increasing cf [DORN](dorn.md)) and LID (linearly increasing cf [Center3D](center3d.md)).
- Architecture
	- Input image $W_I \times H_I \times 3$
	- Frustum feature grid
	- Voxel grid 
	- BEV feature grid
- Frustum feature network
	- D: Depth distribution network, with **direct depth supervision**
		- Input: image feature from block1 of resnet, $W_F \times H_F \times C(=256)$
		- Output: image feature from block1 of resnet, D with size $W_F \times H_F \times D$
	- F: Channel reduction of image feature, with size $W_F \times H_F \times C(=64)$
	- Outer product $G=D \otimes F$
		- Output: Frustum feature grid: G with size $W_F \times H_F \times D \times C$ 
		- Outer product: feature pixels are weighted by depth probability using outer product. This step encapsulates the C channel features in each of the 3D voxel bins (although the 3D voxels bins are still in the form of image + depth bins). 
- Frustum to voxel conversion
	- Trilinear interpolation with known camera intrinsics
	- Input G
	- Output V with size $X \times Y \times Z \times C$
- Voxel collapse to BEV
	- Z*C is stacked and reduced to C with 1x1 convolution. 

![](https://owen-liuyuxuan.github.io/papers_reading_sharing.github.io/3dDetection/res/caddn_arch.png)

#### Technical details
- Voxel size = 0.16 m for mono3D.
- The authors did not do nuScenes as 32 line lidar is too sparse for depth completion.
- How to understand the channels C of frustum feature G and volume feature V? It is like the 3 channels for RGB image. For features, we increased the channels, but the C channel vector is still associated with a 3D location. 


#### Notes
- [code on github](https://github.com/TRAILab/CaDDN)

