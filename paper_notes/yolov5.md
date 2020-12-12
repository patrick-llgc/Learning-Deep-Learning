# [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)

_December 2020_

tl;dr: Pytorch-native improved version of Yolov4.

#### Overall impression
The author of the repo has not released a paper yet, but the repo is highly useful. Many Kaggler commented that its performance is better than [yolov4](yolov4.md). The training speed of Yolov5 is also much faster than Yolov4.

#### Key ideas
- Two great reviews on Zhihu
	- [使用YOLO V5训练自动驾驶目标检测网络](https://zhuanlan.zhihu.com/p/164627427)
	- [深入浅出Yolo系列之Yolov5核心基础知识完整讲解](https://zhuanlan.zhihu.com/p/172121380)
- Focus layer
	- This is a spatial to channel layer that warps H x W x 3 into H/2 x W/2 x12.
	- See [issue in yolov5 github](https://github.com/ultralytics/yolov5/issues/413)
	- Papers such as [TResNet](https://arxiv.org/abs/2003.13630) <kbd>WACV 2021</kbd> and [Isometric Neural Networks](https://arxiv.org/abs/1909.03205) <kbd>ICCV 2019 workshop</kbd>
- Adaptive anchor learning with genetic algorithm

#### Technical details
- Mosaic data aug was first invented in ultralytics's yolov3 and borrowed into [Yolov4](yolov4.md).


#### Notes
- Questions and notes on how to improve/revise the current work  

