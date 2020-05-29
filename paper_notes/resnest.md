# [ResNeSt: Split-Attention Networks](https://arxiv.org/abs/2004.08955)

_May 2020_

tl;dr: A new drop-in replacement for ResNet for object detection and segmentation task. 

#### Overall impression
It is almost a combination of [ResNeXt](resnext.md) and [SKNet](sknet.md), with improvement in implementation (cardinality-major to radix major).

I do feel that the paper uses too much tricks ([MixUp](mixup.md), [AutoAugment](autoaugment.md), distributed training, etc) and is too similar to [SKNet](sknet.md), especially that the hyperparameter selection reduces this work. Engineering contribution > innovation.

#### Key ideas
- Cardinality concept is the same as [ResNeXt](resnext.md).
- The split attention module is very similar to [SKNet](sknet.md) but with the same kernel size.
![](https://pic4.zhimg.com/v2-5124506fd566c147e5763b1b58352f31_1200x500.jpg)
![](https://pic2.zhimg.com/80/v2-70fc8665074b995be4dcc2ec51eecd75_1440w.jpg)
- The change from cardinality-major to radix-major was implemented for better efficiency (how much?).

#### Technical details
- The final selected hyperparameters are K=1 and R=2. This is very similar to SKNet. 

#### Notes
- Analysis of radix-major in [知乎](https://zhuanlan.zhihu.com/p/133805433)
- This work proves that, with tricks, ResNet can also be SOTA. This is better than works reinventing the wheel such as EfficientDet. 
	- MobileNet and DepthWise convolution can only accelerate on CPU and are better suited for edge devices. 
