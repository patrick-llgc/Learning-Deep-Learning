# [SS3D: Monocular 3D Object Detection and Box Fitting Trained End-to-End Using Intersection-over-Union Loss](https://arxiv.org/abs/1906.08070)

_October 2019_

tl;dr: CenterNet like structure to directly regress 26 attributes per object to fit a 3D bbox. 

#### Overall impression
The paper uses a [CenterNet](centernet.md) architecture to regress bounding boxes. The support region is like the Gaussian kernel at the center of the object. The donut region surrounding the kernel is "don't care" region.

The algorithm requires 3D GT in the first place, and requires accurate intrinsics. (KITTI 3D bbox GT is given in camera coordinate, thus extrinsics does not matter.)

SS3D directly predicts 2D and 3D bboxes, similar to [M3D RPN](m3d_rpn.md) and [D4LCN](d4lcn.md).

This paper also demonstrates **the possibility to directly regress the distance of cars from 2D images**. See [youtube videos](https://www.youtube.com/playlist?list=PL4jJwJr7UjMb4bzLwUGHdVmhfNS2Ads_x). This looks quite similar to [Nvidia's drive demo](https://www.youtube.com/watch?v=0rc4RqYLtEU).

#### Key ideas
- The 2D and 3D bounding box is parameterized as 26 numbers. --> Similar to [M3D RPN](m3d_rpn.md) which regresses 12 numbers with YOLO-like structure.
- The 26 numbers are compared with GT, though a L2 norm, weighted by uncertainty. 
- The 26 numbers have different weights, learned through [heteroscedastic uncertainty weighting](uncertainty_bdl.md). The weighted least square is minimized to get the best 3D bbox during inference.
- The 26 numbers can also be trained to fit 3D IoU, but the 26 numbers need to be fitted to a valid 3D bbox online. This requires some complex manipulation of gradient.

#### Technical details
- All pixels in the support (central 20% of bbox) is responsible for detecting the bounding box sizes. Thus NMS is needed to find local optimum. The 26 numbers (from 26 channels most likely) associated with the local optimum point is used to predict the 3D box. 

#### Notes
- Questions and notes on how to improve/revise the current work  

