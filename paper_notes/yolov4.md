# [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)

_May 2020_

tl;dr: The ultimate guide to train a fast and accurate detector with limited resource. 

#### Overall impression
The paper has a very nice review of object detection, including one-stage object detectors, two-stage object detectors, anchor-based ones and anchor-free ones. 

Yolov4 is highly practical and focuses on training fast object detectors with only one 1080Ti or 2080Ti GPU card. Yolov4 runs twice as fast as EfficientDet.

#### Key ideas
- Bag of freebies: Improvements can be made in the training process 
	- Example: data augmentation, class imbalance, cost function, soft labeling etc) to advance accuracy
- Bag of specials: which impacts the inference time slightly with a good return in performance.
	- Example: Plug-in modules such as attention modules, 
- Mosaic data aug: this is very similar to [Sticher](sticher.md). This is similar to increase the batch size.
- Self-adversarial training: first pass modify original image, then second pass train object detection
- Cross minibatch batch norm: improved version of [CBN](cbn.md).
- SAM: Spatial Attention Module from [CBAM](cbam.md) modified to point wise attention.
- [PANet](panet.md): concatenation instead of addition.

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

