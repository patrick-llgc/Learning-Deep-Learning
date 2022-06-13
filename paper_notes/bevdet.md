# [BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View](https://arxiv.org/abs/2112.11790)

_December 2021_

tl;dr: SOTA engineering effort with Lift-Splat-Shoot on BEV detection. Similar to [CaDDN](caddn.md) and [DD3D](dd3d.md).

#### Overall impression
This paper achieves SOTA performance on nuScenes 3D object detection. It uses the SOTA components and did not invent any new modules. The biggest innovation is the proposal of a new data augmentation method in BEV space. 

The BEV detection framework has four components

- image-view encoder: SwinTransformer, ResNet
- view transformer: [LSS](lift_splat_shoot.md)
- BEV encoder: ResNet
- task specific BEV head: [CenterPoint](centerpoint.md).


This work is improved by [BEVDet4D](bevdet4d.md) and [BEVerse](beverse.md).

#### Key ideas
- Multicam BEVDet has much less data samples and thus suffers severe overfitting issues.
- Image space data agumentation: similar to that in [LSS](lift_splat_shoot.md).
- **BEV data augmentation (BDA)**
	- Features in BEV undergoes flipping, scaling and rotating, with corresponding GT undergoing the same augmentaion
	- BDA plays a more important role than IDA in training BEVDet.

#### Technical details
- SOTA image view methods as of late 2021 includes [FCOS3D](fcos3d.md) and [PGD](pgd.md).
- BEV FOV: output space 51.2 m with resolution of 0.8 meters. --> How about in the front?
- Trained with [CBGS](cbgs.md), like in [CenterPoint](centerpoint.md).
- BEVDet exceeded classic lidar based method such as pointPillars.

#### Notes
- [Github](https://github.com/HuangJunJie2017/BEVDet), open sourced in 2022/06.
