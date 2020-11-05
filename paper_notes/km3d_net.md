# [KM3D-Net: Monocular 3D Detection with Geometric Constraints Embedding and Semi-supervised Training](https://arxiv.org/abs/2009.00764)

_September 2020_

tl;dr: Work the geometric reasoning with pseudo-inverse optimization into the neural network.

#### Overall impression
[KM3D-Net](km3d_net.md) is based on the previous work from the same author, [RTM3D](rtm3d.md). [KM3D-Net](km3d_net.md) is highly practical, and works the 3D geometry reasoning module into the neural network to speed things up. Geometric constraint modules in [Deep3DBox](deep3dbox.md), [FQNet](fqnet.md) and [RTM3D](rtm3d.md) are time consuming.

The semi-supervised learning approach is quite interesting and showed that it is possible to get meaningful results just from as few as 500 **labeled** images. Maybe it is a good direction to dig with the self-consistency cues in [UR3D](ur3d.md) and [MoVi-3D](movi_3d.md). The self-supervised learning are done on 

The removal of the depth prediction directly from the neural network makes it possible to do geometric data augmentation and introduce self-supervised loss.

This is the currently the SOTA, much better than previous SOTA [M3D-RPN](m3d_rpn.md).

#### Key ideas
- Architecture
	- Based on [CenterNet](centernet.md) and [RTM3D](rtm3d.md).
	- 1 ch for main center (2D bbox center)
	- 18 = 2 * (8+1) offset from main center
	- 3 chs for dim
	- 8 chs for orientation
	- 1 ch for IoU conf loss 
- Work GRM (geometry reasoning module) into the neural network. 
- **Semi-supervised** training: consensus loss between the prediction of the same unlabeled image under different input data aug. 
- **Keypoint dropout** in the process of training geometry reasoning module. Note that we only need to solve for 3 DoF location, and thus ideally with 2 keypoints we can already recover the 3D bbox. This is confirmed in Ablation study (Fig. 4) that with only two keypoints, the performance can be already reasonable, and no obvious improvements beyond using 4 keypoints. 

#### Technical details
- The prediction of the 9 points are all formulated as **offset regression from the main center instead of heatmaps**. This is different from previous work of [RTM3D](rtm3d.md) where all 9 points are predicted via heatmaps. This paper reasons that heatmap prediction of keypoints are semantically ambiguous and cannot estimate the keypoints of the **truncated** region. --> this makes perfect sense.
- Depth guided L1 loss: initial L1 at near distance, but log when far.

#### Notes
- The self-supervised loss may be used for object detection? Fliplr and add consistency loss.