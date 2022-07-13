# [PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images](https://arxiv.org/abs/2206.01256)

_July 2022_

tl;dr: An improved PETR (with temporal fusion and data dependent PE) for both 3D detection and road layout estimation.

#### Overall impression
This paper explores the joint BEV perception of dynamic vehicles and static road layout. This is similar to [BEVFusion](bevfusion.md) and [M2BEV](m2bev.md).

[BEVFormer](bevformer.md) defines each point on BEV map as one BEV query. The number of BEV query tends to be huge when the resolution of BEV map is relatively large (256x256 = 65536). [PETRv2](petrv2.md) defines a smaller number of (e.g., 256) segmentation queries, each of which predicts the semantic map of the corresponding patch.

#### Key ideas
- Temporal Positional embedding, using 3D coordinate alignment with ego pose.
- Feature guided positional encoder. Different from [PETR](petr.md), 3D PE in PETRv2 is generated in a data-dependent way.
	- The projected 2D features are firstly injected into a small MLP network and a Sigmoid layer to generate the attention weight, which is used to reweight the 3D PE in an element-wise manner.
	- In PETR, the 3D PE is independent with input image.
- Efficient static BEV perception by using large patches
	- Each BEV embedding decoded from one segmentation query is reshaped to a BEV patch (of shape 16x16). This is similar to a channel-to-spatial, or [PixelShuffle operation](https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html).
- The paper also explored the system robustness under typical sensor errors: **extrinsics noise, camera miss, and time delay**.
	- Back camera in nuScenes is most important for KPI as it has 180-deg FOV.

#### Technical details
- [PETRv2](petrv2.md) also observes the regression of tasks performance as compared to single task training. Joint training regresses the static BEV perception more severely. Yet tuning the task weight up (x2) largely eliminates the gap.

#### Notes
- [Code on Github](https://github.com/megvii-research/PETR)
