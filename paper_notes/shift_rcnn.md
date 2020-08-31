# [Shift R-CNN: Deep Monocular 3D Object Detection with Closed-Form Geometric Constraints](https://arxiv.org/abs/1905.09970)

_October 2019_

tl;dr: Extend the work of [deep3Dbox](deep3dbox.md) by regressing residual center positions.

#### Overall impression
The paper has a good summary on mono 3DOD in introduction. 

The geometric constraints become a closed-formed one. This is similar to [deep3Dbox](deep3dbox.md) but slightly different (over-constraint vs exact-constraint).

The idea of [shift RCNN](shift_rcnn.md) and [FQNet](fqnet.md) are quite similar. Both builds on [deep3Dbox](deep3dbox.md) and refines the first guess. But [FQNet](fqnet.md) passively densely sample around the GT and train a regressor to tell the difference to GT, [shift RCNN](shift_rcnn.md) actively learns to regress the difference. The followup work of [FQNet](fqnet.md) is [RAR-Net](rarnet.md) which also actively predicts the offset, but does that iteratively with a DRL agent.

#### Key ideas
- RoiAligned feature to regress 3D orientation and 3D dimension. 
- Optimization to solve for 3D bbox location t'.
- Shift Net work is 2 layer FC network to regress improved final translation of 3D center t''. The input features are t', 2d bbox, dimension, local yaw, global yaw, and camera projection matrix. 
- The volume displacement loss is decomposed into 3 sums of 3 terms, each term is $\Delta x \times h \times w$ and alike. w and h are estimated 3D dimension.

#### Technical details
- They used best IoU to pick the best configuration. This is a bit different from the previous method of picking one that mininizes residual from least square fitting, such as [FQNet](fqnet.md) or [Deep3DBox](deep3dbox.md). This is also used in [MVRA](mvra.md).

#### Notes
- Questions and notes on how to improve/revise the current work  

