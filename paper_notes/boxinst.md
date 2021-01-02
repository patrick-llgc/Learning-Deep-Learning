# [BoxInst: High-Performance Instance Segmentation with Box Annotations](https://arxiv.org/abs/2012.02310)

_January 2021_

tl;dr: Supervising instance segmentation with bbox annotation only. Very close to supervised performance. 

#### Overall impression
The work is based on [CondInst](condinst.md), and only changes the training loss, and are thus equally easy to be deployed in production. Actually this can work with any [single stage instance segmentation](../topic_single_stage_instance_segmentation.md) methods that generates global mask, such as [SOLO](solo.md) and [SOLOv2](solov2.md).

Remarkably, the performance of BoxInst with only bbox supervision can achieve better performnace compared to fully supervised methods such as [YOLACT](yolact.md) and [PolarMask](polarmask.md).

Almost all previous weakly supervised methods uses methods are based on pixelwise mask loss, however this does not work well if we do not have the real mask annotation. The performance looks stunning, and significantly boosted the performance of weakly supervised learning of instance segmentation.

#### Key ideas
- Same network architecture, but diff loss 
	- Mask-box consistency loss: the bbox should tightly bound the mask. 
		- min/max is also differentiable
	- Pair-wise loss on pixel-pixel pairs (edges) with similar colors. 
		- This is based on the observation that similar colors are very likely to have the same label
		- The edge only spans a kxk area (can have dilation, for example, 3x3 with stride 2). The edge must have at least one pixel inside the bbox.
		- Converted to lab space as it is closer to human perception. Modeled by a bell curve. 
- The loss works equally well as pixelwise loss in fully supervised setting. It can benefit from partial mask annotation in a semi-supervised fashion. 


#### Technical details
- The pairwise loss is the key. Even with the mask-box consistency loss, the network may learn to learn the entire bbox as the mask. 
- The weakly supervised method can actually beat the fully supervised [YOLACT](yolact.md) and [PolarMask](polarmask.md).

#### Notes
- [Review of this work by 1st author on Bilibili](https://www.bilibili.com/video/BV1x5411H79b/)
- [Integrated into Github AdelaiDet](https://github.com/aim-uofa/AdelaiDet/)
