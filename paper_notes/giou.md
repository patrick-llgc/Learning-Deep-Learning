# [gIoU: Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630)

_November 2019_

tl;dr: Propose a new metric and loss function based on IoU for object detection.

#### Overall impression
> The optimal objective for a metric is the metric itself.

How to make a metric differentiable and use it as a loss seems to be the trend. This is quite popular in monocular 3D object detection to use 3D IoU as loss.

Dice loss has been used in medical imaging applications for some time now, but it has the issue of zero gradient when overlap is zero.

This seems quite similar to the signed IoU in [monoDIS](monodis.md).

#### Key ideas
- Problem with commonly used l1 or l2 loss for object detection
	- the minimization of loss does not directly correlates with IoU gain.
	- (x, y) and (w, h) does not live in the same space, and thus log transformation is needed
- IoU loss is also scale-invariant (like Dice loss)


#### Technical details
- Summary of technical details

#### Notes
- Compare with sIoU (signed IoU) in [monoDIS](monodis.md).