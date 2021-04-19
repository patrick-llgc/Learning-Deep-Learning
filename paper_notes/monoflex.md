# [MonoFlex: Objects are Different: Flexible Monocular 3D Object Detection](https://arxiv.org/abs/2104.02323)

_April 2021_

tl;dr: Decouple the prediction of truncated objects in mono3D.

#### Overall impression
How to deal with truncated objects remained one key task for mono3D. Most of the existing method (especially [CenterNet](centernet.md)) did not treat this in particular and the prediction on truncated objects looks distorted. This is the first paper which explicitly addresses this challenge. 

The idea is to decouple the learning of truncated objects (outside objects) and untruncated object (inside objects), by using different representative points (or anchor point).

#### Key ideas
- Representative points
	- For inside objects, 3D (projected) center is better than 2D center. --> different from what [RTM3D](rtm3d.md) uses. RTM3D refer to this representative anchor point as "main center". 
	- For outside objects, 3D center is outside the image, and the intersection point between the image edge and the line connecting $x_b$ (2D center) to $x_c$ (3D center).
- Edge fusion
	- Extracts boundary from feature map, 1D conv, then adds back to the feature map
- Depth prediction
	- Direct prediction of transformed target $d = \frac{1}{\sigma(x)} - 1$
	- Depth by scaling predicted keypoints and height (the top surface and bottom surface centers, and two diagonal height)
	- Soft ensemble according to predicted aleatoric uncertainty
- Loss
	- [gIoU loss](giou.md)
	- corner loss, following [monoDIS](monodis.md).

#### Technical details
- The 2D center is not the center of the tight 2D bbox annotated on the image, but the 2D bbox around the partial projected 3D bbox that is inside the image, as can be seen from Fig. 4(c).
- Only random horizontal flip is used for data augmentation. 
- Simply discarding outside objects can improve the performance compared to the baseline, demonstrating the necessity of decoupling outside objects. 

#### Notes
- [Code on github](https://github.com/zhangyp15/MonoFlex)

