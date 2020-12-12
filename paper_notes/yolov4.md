# [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)

_May 2020_

tl;dr: The ultimate guide to train a fast and accurate detector with limited resource. 

#### Overall impression
The paper has a very nice review of object detection, including one-stage object detectors, two-stage object detectors, anchor-based ones and anchor-free ones. 

Yolov4 is highly practical and focuses on training fast object detectors with only one 1080Ti or 2080Ti GPU card. Yolov4 runs twice as fast as EfficientDet.

Overall YOLOv4 is not very friendly to be deployed, given the darknet framework. Two better and more industry-friendly solutions are:

- [PP-YOLO](pp_yolo.md) starts with [YOLOv3](yolov3.md) and uses training tricks to boost performance above [YOLOv4](yolov4.md). 
- [YOLOv5](yolov5.md) is a more engineering friendly, pytorch-native repo.

#### Key ideas
- Great review on Zhihu
	- [深入浅出Yolo系列之Yolov3&Yolov4&Yolov5核心基础知识完整讲解](https://zhuanlan.zhihu.com/p/143747206)
- Bag of freebies: Improvements can be made in the training process 
	- Data augmentation: CutMix and Mosaic
		- photometric, geometric
	- DropBlock regularization: more effective than DropOut for CNN. DropOut was initially proposed for fc layers.
	- Class label smoothing: it actually degrades performance 
	- cIOU loss function
	- CmBN
	- Cosine Anealing LR
	- Dynamic minibatch size. This is similar to [Multigrid](multigrid_training.md)
- Bag of specials: which impacts the inference time slightly with a good return in performance.
	- Plug-in modules such as attention modules
	- SPP
	- SAM
	- PANet
	- activation other than ReLU: Mish (better than Swish, also seen [here](https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/))
	- dIoU-NMS (anchor-free are NMS-free)
	- Random training shapes
- Innovations
	- Mosaic data aug: this is very similar to [Sticher](sticher.md) and [Yolov3 Ultralytics](https://github.com/ultralytics/yolov3). This is similar to increase the batch size.
	- Self-adversarial training: first pass modify original image, then second pass train object detection
	- Cross minibatch batch norm: improved version of [CBN](cbn.md).
	- SAM: Spatial Attention Module (channel wise mean/max pooling) from [CBAM](cbam.md) modified to point wise attention.
	- [PANet](panet.md): concatenation instead of addition.
- Conclusions:
	- ResNeXt50 is better for classification, but DarkNet53 is better for detection
	- When enough trick is used, accuracy does not depend on batch size too much (4 and 8 similar)

#### Technical details
- IOU Loss --> GIOU Loss --> DIOU Loss --> CIOU Loss. See [review on Zhihu](https://zhuanlan.zhihu.com/p/143747206)
- S: eliminating grid sensitivity. Yolov3 uses a sigmoid to regress the relative position inside the grid. The scale factor in yolov4 is [1.2](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg#L973). Essentially this is using the middle range of sigmoid, say stretching [0.2, 0.8] to [0, 1]. See more details in this [PR to openCV](https://github.com/opencv/opencv/issues/17148). When S >> 1, then scale sigmoid approximates L2 loss.
	- See this [issue in yolov4 repo](https://github.com/AlexeyAB/darknet/issues/3293) for background of this discussion.
- Most effective tricks:
	- Mosaic data aug
	- GA (genetic algorithm) to use 10% of training to finetune parameters.
	- Cosine anealing
- SqueezeNet, MobileNet and ShuffleNet are more friendly for CPU but not GPU.
- SPP: in the original paper [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729) <kbd>TPAMI 2015</kbd> converts an image to a fixed length one-dimensional feature vector, and SPP is used for object detection. Yolov3 improves it by concatenating the output of max pooling with stride 1. This helps increases the receptive field fast. 
- FLOPS vs inference time: SE on GPU usually increases inference time by 10%, although flops only increases 2%. SAM from [CBAM](cbam.md) does not increase inference time at all.
- [CSPNet: A New Backbone that can Enhance Learning Capability of CNN](https://arxiv.org/abs/1911.11929): Cross stage partial network splits feature channels into two groups: one group passes through conv layers and the rest keeps the same. 

#### Notes
- Why not regressing the position inside the grid [0, 1] directly with L1 loss?

