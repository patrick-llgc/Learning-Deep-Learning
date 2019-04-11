# [MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving](https://arxiv.org/pdf/1612.07695.pdf)

_Mar 2019_

tl;dr: Combines object detection, scene classification and road segmentation into one combined net.

#### Overall impression
The paper comes form Raquel Urtasun's group, and is one of the best paper with codes for kitti's road detection. Encoder + 3 task specific decoders for multitask joint training and inference. The best contribution of this paper may be the light weight detector network YOLOv1 as a light weight RPN.

Btw the writing is surprisingly bad with numerous typos.

#### Key ideas
- Two-stage proposal methods have size-adjustable features. This scale invariance is the main advantage of proposal based systems.
- For classification of images having objects prominently centered in the image, a small size may be good enough. For complex scene understanding, the authors argued that a higher resolution is necessary.
- For segmentation, a standard FCN is used. The weight is initialized to a bilinear interpolation, and the skip connection is initialized to small weight. These leads to improvement in training process.
- BottleNet (1x1) layer is used to reduce dimension and computation.
- The three heads have different training schedule.
- Detection framework:
	- The image is divided into a grid (32x32 pixels). Each grid is responsible for detecting if it intersects with at least one bbox and regresses to the bbox whose center is closed.
	- $c_x = (x_b - x_c) / w_c, c_w = w_b/w_c$. **There is no anchor box. This is essentially YOLOv1**
	- $L = \delta c_p (\sum_{i\in \{x, y, w, h\}}|c_i - c'_i|)$
	- The initial $H \times W \times 6$ prediction is then used to perform ROIAlign on the original image scale. The ROIAligned features and the prediction from the first stage are concatenated to predict the offset. 


#### Technical details
- A fully connected layer can be viewed as a 1x1 in the case where the spatial resolution is already 1x1 (already squashed by fc). [LeCunn's comments](https://www.facebook.com/yann.lecun/posts/10152820758292143) are misleading. Check out the [FCN](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) paper for a better context.

#### Notes
- This basically mimics the faster/mask-RCNN framework. The authors did not perform ablation study to reveal why it speeds up the inference. My guess is that it removed the use of bounding box. Are bounding boxes really necessary?
	- Maybe not. See recent studies [Feature Selective Anchor-Free Module for Single-Shot Object Detection](https://arxiv.org/pdf/1903.00621.pdf) and [FoveaBox: Beyond Anchor-based Object Detector](https://arxiv.org/pdf/1904.03797v1.pdf).
	- The original YOLO paper is extremely easy to read, and there is no anchor box. YOLOv2 introduces anchor box.
- Code available at [github](https://github.com/MarvinTeichmann/MultiNet).