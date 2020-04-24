# [BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation](https://arxiv.org/abs/2001.00309)

_April 2020_

tl;dr: Learn a low-resolution attention map to improve [YOLACT](yolact.md). 

#### Overall impression
The paper improves upon [YOLACT](yolact.md). It is not only using bbox to crop the prototype map but also predict an attention map within the bbox. From this standpoint, it is more similar to Mask RCNN than even [YOLACT](yolact.md). 

[YOLACT](yolact.md) predicts just a single number to blend prototype masks, but [BlendMask](blendmask.md) predict a low-res attention map to blend the masks within the bbox. 

[BlendMask](blendmask.md) is trying to blend with a finder grained mask within each bbox, while [CondInst](condinst.md) is trying to blend with deeper convs with dynamically predicted filters. 

#### Key ideas
- K is the number of bases (prototypes in YOLACT). Each feature map location predicts K x M x M masks, realized by a KMM sized channel. M = 1 for YOLACT. M is typically quite small with max of 14.
- The feature maps (bases) are RoIAligned with bboxes, before multiplying with attention maps in the blender module. 
- BlendMask actually works with only L=1 base. K=4 has best tradeoff.
- Compare with Mask RCNN, BlendMask moves computation of R-CNN head before the RoI sampling to generate position sensitive feature map. Repeated RoI-based mask representation learning is avoided. 
	- increasing mask resolution will lead to quadratic speed penalty
	- increasing number of object will lead to linear speed penalty
- The position sensitive feature map is related to [R-FCN](rfcn.md) and [FCIS](fcis.md).

#### Technical details
- Removing anchors enables heavier duties to the head such as predicting an attention map.
- The best model is 56 x 4 x 7 = R x K x M. 
	- R: bottom level RoI resolution
	- K: number of bases
	- M: top level attention map


#### Notes
- What I don't understand is why the feature map of each object has to be fixed to RxR?


