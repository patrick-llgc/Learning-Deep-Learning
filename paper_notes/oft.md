# [OFT: Orthographic Feature Transform for Monocular 3D Object Detection](https://arxiv.org/pdf/1811.08188.pdf)

_July 2019_

tl;dr: Learn a projection of camera image to BEV for 3D object detection.

#### Overall impression
This paper is brilliant! It combines several key innovations in the past year: camera to BEV projection (similar to [pseudo-lidar](pseudo_lidar.md)), and anchor-free object detection (similar to [CenterNet](centernet.md)).

However the way of reprojection without depth estimation perhaps limited the performance of the model, which is significantly below that of [MLF](mlf.md) and [pseudo-lidar](pseudo_lidar.md). For simple depth estimation and 3D reasoning using 2bbox and CV, refer to [Deep3dBox](deep3dbox.md) and [MonoPSR](monopsr.md).

The OFT transformation network inspired future work of [PyrOccNet](pyr_occ_net.md) for monocular bev semantic map prediction.

The network does not require explicit info about intrinsics, but rather learns the constant mapping. That is why extensive augmentation was required to do this. --> why not injecting intrinsics implicitly?

#### Key ideas
- [CenterNet](centernet.md)-like Detection pipeline. 
	- regress object center
	- regress center offset, size and orientation (cos and sin)
- NMS of object center in confidence map to find local maximum. 
- The topdown network is actually quite important, as shown in the ablation study. --> This is due to the ambiguity in reprojecting the 2D camera image pixels to 3D space, and the topdown network is use to "reason in 3D". However I do feel that estimating a depth network explicitly before reprojecting to 3D may be much better. Maybe by that time the topdown network can be removed entirely and reduce to a simple ROIpooling. Or a shallow network can be used to adjust the depth.

#### Technical details
- Replace batchnorm with groupnorm.
- Data augmentation and adjusting intrinsic parameters accordingly (including cx, cy and fx and fy, c.f., [depth in the wild](learnk.md) paper).
- Sum loss instead of averaging to avoid biasing toward examples with few object instances.

#### Notes
- No occlusion was accounted for during 3D transformation. That means many voxels in 3D space actually share the same features pooled from the 2D feature map. --> This can be improved by estimating depth before reprojection.
