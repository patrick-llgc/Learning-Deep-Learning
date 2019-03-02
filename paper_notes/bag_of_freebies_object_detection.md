# [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf)

_Mar 2019_

tl;dr: Commonly used tricks to boost object detectors (both one and two-stage pipelines). Cosine learning rate, class label smoothing and mixup is very useful. Half-half mixup gives the largest performance boost.

#### Overall impression
Cosine learning rate, class label smoothing and mixup is very useful. It is surprising that mixup technic is useful in object detection setting. (Need more investigation into this topic)

#### Key ideas
- Mixup helps in object detection. However 0.5:0.5 ratio works better than 0.1:0.9 mixup ratio. 
	- Random sampling from beta-distribution is slightly better than 0,5:0.5 fixed even mixup.
	- Object detector trained with mixup is more robust against alien objects ("elephant in the room" test)
- Data augmentation 
	- It is especially important in the context of SSD in order to be able to detect objects at different scales
	- Data augmentation has minimal effect on multi-stage pipeline. The authors argue that the roi pooling ops on feature maps substitute the op of random cropping and therefore does not require extensive data augmentation. [The SSD paper](https://arxiv.org/pdf/1512.02325.pdf) also mentioned that Faster RCNN may benefit less from data augmentation as they are relatively robust to object translation by design.
	- Note that when dataset is small, we should [still use data augmentation](https://www.datacamp.com/community/tutorials/object-detection-guide).

#### Technical details
- Label smoothing: cross entropy and softmax encourages the model to be too confident in its predictions and is prone to overfitting.
$$q_i' = (1-\epsilon) q_i + \frac{\epsilon}{K}$$
where K is the num of classes.


#### Notes
- [目标检测任务的优化策略tricks](https://zhuanlan.zhihu.com/p/56792817)
- Why Mixup works?