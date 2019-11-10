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
- Compare with sIoU (signed IoU) in [monoDIS](monodis.md)
- gIoU
![](https://pic2.zhimg.com/80/v2-a60505e06dc14bac8fc9d4030e525609_hd.jpg)
![](https://pic3.zhimg.com/80/v2-3dcaecc3c0edc6a78985c2c9a93d7216_hd.jpg)
- The sIoU
![](https://pic3.zhimg.com/80/v2-023352d72006fe9638fb9b7d7844e096_hd.jpg)
![](https://pic2.zhimg.com/80/v2-7a4288ce85acb4c859e8e5bc12e53769_hd.jpg)

#### Notes
- [VNet](https://arxiv.org/abs/1606.04797) in 2016 was the first to propose Dice Loss in image segmentation. [Lovasz Softmax](https://arxiv.org/abs/1705.08790) (CVPR 2018) is a high-performing surrogate for IoU loss, but is also used for segmentation. [Unit Box](https://arxiv.org/abs/1608.01471) is the first attempt to use IoU loss in object detection.