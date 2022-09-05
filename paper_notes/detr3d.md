# [DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries](https://arxiv.org/abs/2110.06922)

_November 2021_

tl;dr: BEV object detection with DETR structure. 

#### Overall impression
Inspired by [DETR](detr.md), the paper uses sparse queries in BEV space for BEV object detection. It manipulates prediction directly in BEV space. It does not rely on dense depth prediction and avoids reconstruction errors. It is in a way similar to [STSU](stsu.md).

Mono3D methods have to rely on per-image and global NMS to remove redundant bbox in each view and in the overlap regions.

The work is further improved with 3D positional embedding by [PETR](petr.md) and [PETRv2](petrv2.md).

> The extension of [DETR3D](detr3d.md) to temporal domain is relatively straightforwad, using the 3D reference point, transforming to the past timestamps using ego motion, and then project to the images from the past timestamps.

#### Key ideas
- Iterative refinement of object queries.
	- The iterative refinement process is similar to [Sparse RCNN](sparse_rcnn.md) and Cascade RCNN. 
	- Predicts the bbox centers, project centers back to images with IPM, and sample feature points with bilinear interpolation and integrate them into queries.
	- It also works with only a single pass, but with iterative refinement the results gets better. L=6 in this paper. 
- Performance are much better in overlap regions where objects are more likely to be cut-off.

#### Technical details
- Initialization seems to matter quite a lot for transformers based networks. If the network is pretrained from FCOS3D, the performance is boosted by 5 abs points (See table 1).
- Ground-truth depth supervision yields more realistic pseudolidar point cloud than self-supervised depth. That is perhaps the reason why [DORN](dorn.md) instead of [Monodepth2](monodepth2.md) pretrained weights are preferred for pseudo-lidar papers. 
- DETR3D also predicts velocity, thus 7+2=9 DoF bbox. --> But why? The predicted velocity must be unreliable due to lack of a memory module. 
- AdamW is the optimizer. 

#### Notes
- This paper uses sparse transformation instead of transforming the entire feature to BEV. This is what Andrej Karpathy mentioned the next step of FSD in his Tesla AI day talk 2021. 

