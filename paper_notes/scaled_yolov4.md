# [Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036)

_November 2020_

tl;dr: Best practice to scale single-stage object detector. [EfficientNet](efficientnet.md) for [Yolov4](yolov4.md).

#### Overall impression
The paper is not as well written as the original [Yolov4](yolov4.md) paper. This paper follows the methodology of [EfficientNet](efficientnet.md). 

From this [review on Zhihu](https://www.zhihu.com/question/430668054/answer/1580560177) it looks like Scaled-YOLOv4 is heavily based on [YOLOv5](yolov5.md).

#### Key ideas
- When input image size is increased, we must increase depth or stages of the network. Best practice is to follows the steps:
	- scale up the "size of image + #stages"
	- scale up depth and width according to required inference time
- Once-for-all network
	- Train one Yolov4-Large network, and drop the later stages for efficiency during inference. 

![](https://pic2.zhimg.com/v2-6665b587c4a83a2f4a85fb37bd2a2f57_r.jpg?source=1940ef5c)

#### Technical details
- It uses OSA (one shot aggregation) idea from VoVNet. Basically instead of aggregating/recycling features at every stage, OSA proposes to aggregate the features only once at the end. [source](https://paperswithcode.com/method/vovnet)
![](https://paperswithcode.com/media/methods/Screen_Shot_2020-06-23_at_3.46.14_PM_5fzw8NV.png)

#### Notes
- Code on [github](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-large)

