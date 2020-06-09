# [End-to-End Lane Marker Detection via Row-wise Classification](https://arxiv.org/abs/2005.08630)

_June 2020_

tl;dr: Row-wise pixel **classification** of lane detection from Qualcomm Korea.

#### Overall impression
This is inspired by GM Isreal's drivable space prediction method [StixelNetV2](stixelnetv2.md). 

It translate the lane detection problem into a row-wise classification task, which takes advantage of the innate shape of the lane markers (one x value per y) (which is just the opposite for DS prediction, one y value per x, as shown in [StixelNetV2](stixelnetv2.md)).

Pixel segmentation task will require postprocessing algorithm such as pixel wise clustering algorithm.

#### Key ideas
- The network squeezes the feature map via a sequence of HRM (horizontal reduction module) to reduce the x dim to 1. In HRM, there are two parallel branch, one normal horizontal x-pooling branch, and one spatial to channel and then 1x1 channel reduction branch. Then the two are followed by [SE-Block](senet.md). This is also called horizontal pixel unshuffle layer. This is the reverse pxiel shuffle operation as in [Subpixel convolution](subpixel_conv.md).
- The prediction of x pixel did not use regression but rather uses a row-wise classification layer. This is a repeated pattern we have seen in neural network that **classification works better than regression** (including keypoint detection, depth regression, anchor-free object detection, etc). A simple CE works well enough. 

#### Technical details
- In face landmark detection, a KL loss based on laplacian distribution is usually used. ([Laplace Landmark Localization](https://arxiv.org/abs/1903.11633) <kbd>ICCV 2019</kbd>).
- Training with the upgraded AdamW for training, similar to [DETR](detr.md).

#### Notes
- This seems to be what most industry players are doing for lane detection. See [3D LaneNet](3d_lanenet.md).

