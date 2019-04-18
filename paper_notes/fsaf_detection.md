# [FSAF: Feature Selective Anchor-Free Module for Single-Shot Object Detection](https://arxiv.org/pdf/1903.00621.pdf) 

_April 2019_

tl;dr: Introduced anchor-free module on top of FPN, which can be used alone or with anchor-based method (such as RetinaNet). The paper also proposes to select feature layers automatically instead of manually assign target according to object size.

#### Overall impression
FSAF has two main contribution: FS (online/learned feature selection) training method and AF (anchor free) path. It would be of interest to see how would RetinaNet improve just based on the FS method vs the original heuristic/manual feature selection. Also, how about combining the feature maps (like in [Panoptic FPN](panoptic_fpn.md)) so that there is no need to do feature selection?

#### Key ideas
- In RetinaNet, each level of the pyramid is used for detection objects at a different scale. 
- Anchor free (AF): encode the instances in an anchor-free manner to learn the parameters for cls and reg.
- The classification subnet predicts the probability of objects at each spatial location for K classes. 
	- In training, we encode each GT bbox with a dilated box (effective box) at the center and ignore a donut region (ignoring box - effective box) for loss aggregation. This is done for each scale of the pyramid. This is like the gaussian mask used in [CornerNet](cornernet.md) and [CSP](csp.md).
	- Focal loss is used for non-ignoring region
- The bbox regression subnet: 4-channel feature map, each predicting the offset from that pixel to the bbox edges. 
	- IoU loss is used
	- **Patrick's note**: Each pixel has to be assigned a bbox. If the pixel lies in the effective box of a bbox, then the pixel is assigned to the bbox. If the pixel falls under effective box of multiple bboxes, then the smallest bbox is selected.
- Online Feature Selection (FS): the loss from each level of the feature map is selected and **the level generating the min loss** is used for backprop. The reason is to pull down the lowerbound as much as possible. In inference, no selection is needed as results from multiple level is aggregated and NMS'ed.
- It achieves SOTA among single stage object detectors. It is 44 AP on COCO, better than 43 AP [ExtremeNet](extremenet.md) and 42 AP [CornerNet](cornernet.md). This is surpassed by 47 AP [CenterNet](centernet_cas.md).

#### Technical details
- If effective box of two instances overlap in one level, then the smaller instance has higher precedence.

#### Notes
- The paper has one of the best description of anchor boxes I have seen. 

> **Anchor boxes** are designed for discretizing the continuous space of all possible instance boxes into a finite number of boxes with predefined locations, scales and aspect ratios. 

- I am not totally convinced that the way to deal with the occlusion (in particular, **concentric occlusion** where the effective box of two objects overlap) of the same class objects is handled properly. However we do not know occlusion is one the major failure mode. COCO seems to be fine in that regard. 
- Batch size of 1 is used for inference. This is a good approximation for model deployment in most cases. 