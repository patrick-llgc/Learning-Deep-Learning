# [Light-Head R-CNN: In Defense of Two-Stage Object Detector](https://arxiv.org/pdf/1711.07264.pdf)

_April 2019_

tl;dr: Faster than two-stage detectors and more accurate than one-stage detectors.

#### Overall impression
The paper analyzed the computation burden in Faster RCNN and R-FCN, and proposes a more balanced network. The authors fine-tune-fu is amazing.

It is now possible to integrate FPN into R-FCN with the changed architecture of light head RCNN.

The PS RoIPooling is replaced with PS RoIAlign. This RoI Align technique also improved AP by more than 1 point. -->  PS RoIAlign is further extended to rotated PS RoIAlign in [RoI transformer](roi_transformer.md).

#### Key ideas
- Faster RCNN has a heavy head (with two fc layers), and R-FCN has a heavy score-map. Even if the base network can be reduced, no much improvement will be gained. 
	- Faster RCNN's computation grows when more proposals are needed. There is a global avg pooling and 2 fc layers. 
	- R-FCN produces a very large score map with #classes x p x p. This way, the head is cost-free.
	- R-FCN cannot leverage FPN directly due to large memory consumption, if we want to consume the high resolution feature maps. 
- Two main modifications:
	- large kernel separable convolution to convert the feature map from the backbone to a thin score map. We can use C_mid to control the complexity of computation.
	- Reduced score map channel from #classes x p x p to 10 x p x p. (This reduction only holds when #classes >> 10). As 10 is not necessarily the #classes, so we ned to have a fc layer for final prediction.
- The pooled feature map has only 10 channels.
- With light weight backbone such as Xception, it can achieve ~100 fps inference speed.

#### Technical details
- The feature map for COCO is reduced from 3969 (7x7x81) to 490 (7x7x10)

#### Notes
- For special applications like vehicle or pedestrian detection, it perhaps does not save too much as #classes is small (1 or 2).
- In CV community the majority of effort is to have a good-performance generalized object detector. However in real-world industrial applications, we need good performing detectors focusing on specific object classes, most likely with limited computational resource and time for inference. 
