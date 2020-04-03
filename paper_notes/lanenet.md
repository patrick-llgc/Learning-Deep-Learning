# [LaneNet: Towards End-to-End Lane Detection: an Instance Segmentation Approach](https://arxiv.org/abs/1802.05591)

_March 2020_

tl;dr: Binary segmentation and learning of embedding for instance segmentation, plus learned perspective mapping (adaptive IPM). 

#### Overall impression
The paper proposed a good method to perform **instance segmentation on long/thin objects** as conventional detect and segment pipeline does not work well (they are better suited for compact objects). Binary segmentation + embedding for clustering into instances. 

The clustering idea directly inspired [Semilocal 3D Lanenet](semilocal_3d_lanenet.md) to cluster image tiles together which the same lanes pass through. 

The idea to predict vanishing point to guide laneline detection is similar to [VPGNet](vpgnet.md), but LaneNet is not predicting a point but rather directly predicting the homographic transformation. 

The HNet is trained separately. This **decoupled** design makes the system scalable. Similar to the idea of the 3DGeoNet in [Gen-LaneNet](gen_lanenet.md).

#### Key ideas
- Lane fitting with low order polynomial is usually done in BEV space. Fitting directly in perspective space leads to inaccuracies (shown in Table III). Fixed perspective mapping cannot handle uphill or downhill images very well, and thus we need a learned mapping.
- **What is a good homography transformation (IPM)**? The paper defines that as one mapping that minimizes reprojection errors of fitted curves. This is trained with a lightweight network (called H-Net) to predict a conditional Homography matrix (6 DoF to keep horizontal lines horizontal, assuming zero roll). 
	- The projection-fitting-reprojection pipeline is differentiable as there is a **closed-form solution** to this. Autograd can be used for back-propagation.

#### Technical details
- The "lanes" in this paper actually refers to "lane lines" in other literature. 
- The architecture backbone is with the real-time segmentation network of [ENet](https://arxiv.org/abs/1606.02147).

![](https://miro.medium.com/max/488/1*NS54REfUOdLyrQJHz5TkWQ.png)

#### Notes
- [github repo in tensorflow](https://github.com/MaybeShewill-CV/lanenet-lane-detection)
