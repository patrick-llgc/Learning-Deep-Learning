# [Probabilistic two-stage detection](https://arxiv.org/abs/2103.07461)

_March 2021_

tl;dr: How to build fast and accurate two stage detector on anchor-free one-stage detectors. 

#### Overall impression
It seems to me that the "probabilistic" looks fancy, but actually are not that appealing for practical use. The main achievement of this paper is on how to extend SOTA one-stage detectors for two stage and make them fast.

First stage of most two stage detectors is to maximize recall. 

Current SOTA one stage models rely on heavier separate cls an reg branches than two-stage models. They are no longer faster than two stage methods if the cls (vocabulary) is large. 

Previous two stage methods are slow due to the large number of proposals in the first stage. And two-stage methods 

#### Key ideas
- Architecture
	- 1st stage: Class-agnostic one-stage detector predicts object likelihood. Only one class object detection.
	- 2nd stage: predicts classification conditioned on a detection (the bbox refinement is kept the same as Faster-RCNN)
- The speed up due to the drastic reduction in the number of classes more than makes up for the additional costs of the second stage.
- The main difference between [CenterNet2](centernet2.md) and classifical two-stage object detectors
	- First stage maximizes object likelihood instead of recall
	- Overall classification score is conditioned on the class-agnostic detection score of the first stage in CenterNet2.

	

#### Technical details
- One stage detectors define positive and negative samples differently. Anchor-based method use IoU overlap to determine positive cases, and anchor-free method use locations to determine positive cases. See [ATSS](atss.md) for a review. 
- CenterNet*: upgraded CenterNet
	- FPN, and assigning GT to diff FPN levels. Original centerNet works on a single scale.
	- distance to bbox as regression target (following [FCOS](fcos.md))
	- gIoU loss for bbox reg, following [gIoU](giou.md).
- CenterNet2: cascadeRCNN style multi-stage with CenterNet* as first stage.

#### Notes
- The idea of having a class-agnostic first stage aligns with my observation that the fewer classes we have, the better performance we have with CenterNet.

