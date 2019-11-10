# [IoUNet: Acquisition of 	Localization Confidence for Accurate Object Detection](https://arxiv.org/abs/1807.11590)

_November 2019_

tl;dr: Regress a separate branch to estimate the quality of object detection in terms of IoU. 

#### Overall impression
The vanilla version of IoU-Net (with the prediction of IoU and Precise RoI Pooling) is already better than baseline, most likely due to the regularization effect of the IoU branch. 

> The classification confidence indicates the category of a bbox but cannot be interpreted as the localization accuracy.

It generates better results than SoftNMS (decrease the scores of overlapping ones instead of eliminating the overlapped candidate, see [Review of Soft NMS on Zhihu](https://zhuanlan.zhihu.com/p/51654911)), and can be dropped in many object detection frameworks. 

[KL Loss](kl_loss.md) and [IoU Net](iou_net.md) are similar, but are different in implmentation. KL loss directly regresses mean and var from the same head, instead of a separate head for IoU prediction in IoU Net. Also Var Voting is one forward pass, not like the IoU Net's iterative optimization.

#### Key ideas
- Conventional NMS is ignorant of the localization accuracy, while the classification scores are typically used as the metric for ranking proposals.
- localization quality is non-monotonic in iterative bounding box regression.
- **IoU-guided NMS**: keep the box with the highest IoU prediction, rather than cls confidence.
- **PrRoIPooling**: Use integration to make the pooling operation continuous wrt bin locations. This is an improvement over RoIAligna and RoIPool. RoIAlign only considers 2x2 point inside the bbox for calculating the feature map, but PrRoI Pooling removes this quantization altogether.
- **iterative optimization** using the score estimator as a judge. Precise RoIPooling to make the backpropagation possible. 
- The regressed iou score correlates with IoU better than cls confidence
![](https://blog-1258449291.cos.ap-chengdu.myqcloud.com/Blog/IOU-Net/1547292464564.jpg)

Classification confidence tends to be over-confident and is bipolar. 
This is similar to the results in [gaussian yolov3](gaussian_yolov3.md).

![](https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_225489%2Fproject_350848%2Fimages%2Fx4.png)

#### Technical details
- The IoU prediction branch is trained with random jittering online data aug.

#### Notes
- The idea of using a head to regress the IoU is very much like that of [FQNet](fqnet.md), although FQNet aims to regress the 3D IoU from overlaid wireframe. 
- Iterative optimization solved the problem raised by Cascade RCNN that iterative bbox regression does not improve beyond 2 stages. 
- Also iterative optimization should be able to apply to [FQNet](fqnet.md) as it has a predictor to tell the 3D IoU.