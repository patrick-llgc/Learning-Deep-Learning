# [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arxiv.org/abs/2006.04388)

_October 2020_

tl;dr: Generalized focal loss that can optimize for any continuous number and distribution. 

#### Overall impression
The paper follows the method of [focal loss](focal_loss.md) (modulating Cross Entropy by L2 loss). Actually cross entropy can be easily extended to regressing any number between 0 and 1, but it just have a very flat bottom. Now generalized focal loss modulates this extended cross entropy by L2 loss. 

A recent trend in one-stage detector is to introduce an individual prediction branch to estimate the quality of localization. The center-ness ([FOCS](fcos.md) and [ATSS](atss.md)) or IoU score branch ([IoUNet](iou_net.md)) can be trained separately and used in NMS process. But the training and test is decoupled. Even worse, the negative bbox does not have IoU supervision and can have extremely high IoU predictions and thus degrades the NMS process. 

Bbox boundaries are generally formulated as a Dirac delta function (deterministic) or Gaussian ([Gaussian yolo](gaussian_yolo.md) and [KL Loss](kl_loss.md)). This paper targets to formulate the boundary as an arbitrarily shaped distribution. This formulation itself reaches the same performance as baseline, but with DFL (distributional focal loss), it is better. --> for loss on a distribution, cf [Unsuperpoint](unsuperpoint.md).

#### Key ideas
- QFL (quality focal loss): Unifies classification score and IoU quality to be one **cls-iou score**. The target is dynamically updated online and it is in (0, 1]. For negative samples, the target is 0.
- DFL (distributional focal loss): Directly optimizes a distribution of bbox boundaries. the regression target is quantized into n (n=14) bins. The target is expressed as the integral over the distribution. 
	- [gIoU loss](giou.md) actually should work quite well. In order to further boosts the performance, 2 nearest bins from the target are selected from the n targets after a softmax layer is used to calculate the loss.

#### Technical details
- Bbox classification: why multiple sigmoid vs softmax? [FCOS](fcos.md)

#### Notes
- [review by first author on Zhihu 知乎](https://zhuanlan.zhihu.com/p/147691786)
- How is n selected? --> in the above review post, and in Fig. 5(c), it is 0 to 16 in ATSS, 0 to 8 in FCOS. The delta is selected to 1 after ablation study. 