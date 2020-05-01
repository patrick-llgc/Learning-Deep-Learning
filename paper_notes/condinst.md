# [CondInst: Conditional Convolutions for Instance Segmentation](https://arxiv.org/abs/2003.05664)

_April 2020_

tl;dr: Learn dynamic filters to manipulate prototype masks. Sibling to [SOLO](solo.md).

#### Overall impression
The author is also the creator of [FCOS](fcos.md). Both [SOLO](solo.md) and [CondInst](condinst.md) builds on top of FCOS anchor-free object detector. 

The paper extends the idea of [YOLACT](yolact.md) and [BlendMask](blendmask.md) of manipulating high resolution prototype masks to generate instance masks. This solves the mask bleeding problem. 

With depth=1 filter, [CondInst](condinst.md) is essentially [YOLACT](yolact.md) in predicting linear combination of the prototype masks. The AP 30.9 is similar to YOLACT's 31.2 as well.

The dynamic filters encodes the masks implicitly. It can be seen as a data dependent decoder. This is to be compared with a fixed decoder in [Mask Encoder Inst.](meinst.md).

The dynamic filter idea is used in SOLOv2. Matrix NMS is also useful to CondInst as well.

#### Key ideas
- Drawbacks of Mask RCNN
	- Axis-aligned bbox, which may contain excessive irrelevant features. 
	- Heavy head
	- Fixed resolution masks for all object sizes. 
- Instance segmentation requires different output for objects with the same appearance at different locations. Mask RCNN achieves this by explicitly encoding location information in RoIAlign. [SOLO](solo.md) achieves this by assigning channels to grids. 
- Mask RCNN represents instances by bbox. **CondInst represents objects as dynamic filters.** <-- This is similar to the implicit representation of 3D in [DVR: Differentiable Volumetric Rendering: Learning Implicit 3D Representations without 3D Supervision](https://arxiv.org/abs/1912.07372).
- Even one prototype mask will work reasonably well!

#### Technical details
- Only 169 variables in the 3 dynamic filters. 
	- C maps + 2 coordConv maps = 10
	- First layer: 10 x 1 x 1 x 8 + 8 = 88
	- Second layer: 8 x 1 x 1 x 8 + 8 = 72
	- Third layer: 8 x 1 x 1 x 1 + 1 = 9 
	- Total = 88 + 72 + 9 = 169
- Prototype masks are output from C3, thus 1/8 orig resolution. Upsampling by four brings it to half orig resolution. This is critical to good performance. 

#### Notes
- [Video by Zhi Tian](https://www.techbeat.net/talks/MTU4ODAzMDQ2ODM5NS0zODAtNjk1Mzc=)

