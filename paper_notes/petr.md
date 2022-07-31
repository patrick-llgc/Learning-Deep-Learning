# [PETR: Position Embedding Transformation for Multi-View 3D Object Detection](https://arxiv.org/abs/2203.05625)

_July 2022_

tl;dr: Improvement of DETR3D with 3D positional embedding.

#### Overall impression
[DETR3D](detr3d.md) provides an intuitive solution for end-to-end 3D object detection. However there are several issues. The predicted coordinates of reference point may not be that accurate, making sampled features out of object region. Also the online 2D-to-3D transformation and feature sampling will hinder online deployment. 

The object query initialized from 3D space can directly perceive 3D object information by interacting with the produced 3D position-aware features. 

![](https://cdn-images-1.medium.com/max/1600/1*IiSaLdbJhlXGzdyJBdm2cw.png)

PETR believes that using explicit 2D-3D projection during feature transformation will hinder network's capability to perform global reasoning (Note: which I do not necessarily agree), thus it **breaks the explicit reprojection operation**. Instead it uses 3D positional embedding to facilitate global reasoning (3D position-aware features), and ask the neural network to learn **implicitly** where to look by supplying the 2D images with 3D positional embedding. **With this embedding, 2D regions corresponding to the same 3D region will have similar 3D embedding.**

The PETR idea resembles [CoordConv](coord_conv.md) and [CamConv](cam_conv.md), but instead in 2D, this time the positional embedding is in 3D. In a way, [PETR](petr.md) can be seen as an extension of [CamConv](cam_conv.md). [PatchNet (rethinking pseudo-lidar)](patchnet.md) also explores this topic regarding how to represent the 3D information with 2D conv more efficiently: providing 3D information (x, y, z) is more effective than RGBD.

#### Key ideas
- PETR proposed the idea of **3D position-aware features**. The idea is to encode the 3D coordinates effectively with 2D features. 
	- For each pixel, generate D depth bins.
	- In each voxel of WxHxD volume, get the (X, Y, Z, 1) in camera coord, and transform to rig coord.
	- Blend 4D channels to C channels to match with camera image feature channel
	- Blend with addition or concatenation.
- [PETR](petr.md) converges slower than [DETR3D](detr3d.md). The authors argue that PETR learns the 3D correlation through global attention while DETR3D perceives 3D scene within local regions (with the help of explicit 3D-2D feature projection).

#### Technical details
- The authors argue that in [DETR3D](detr3d.md) only the image feature at the projected point will be collected, which fails to perform the representation learning from global view. --> Actually this may not be that of a big issue for BEV perception, especially for object detection, which requires very localized attention. **I would rather consider this as an advantage** of [DETR3D](detr3d.md) and methods alike, such as [BEVFormer](bevformer.md). --> Maybe adding this 2D-3D explicit link will boost the performance even further, with faster convergence?
- The parameter settings in many of the experiments does not matter that much. For example, Table 4 ablation study is not necessary, in particular the Z range of -10 to 10 meters.
- In Fig.3, the FC seems to stand for "fully convolutional". It is actually chosen to be 1x1 in the ablation study in Table5. **What is surprising is that if 3x3 is used instead of 1x1 in the feature blending, the network cannot converge.** --> The authors argue that this breaks the correspondence between 2D feature and 3D position. This is fishy.

#### Notes
- The attention-based visualization is very interesting. --> We should do this as well for our own experiment.
- [Code on github](https://github.com/megvii-research/PETR)
