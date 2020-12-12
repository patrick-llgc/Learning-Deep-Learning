# [PP-YOLO: An Effective and Efficient Implementation of Object Detector](https://arxiv.org/abs/2007.12099)

_December 2020_

tl;dr: A bag of tricks to train [YOLOv3](yolov3.md).

#### Overall impression
This paper and [YOLOv4](yolov4.md) both starts from [YOLOv3](yolov3.md) but adopts different methods. YOLOv4 explores extensively recent advances in backbones and data augmentation, while PP-YOLO adopts more training tricks. Their improvements are orthogonal.

The paper is more like a cookbook/recipe, and the focus is how to stack effective tricks that hardly affect efficiency to get better performance.

#### Key ideas
- Bag of training tricks
	- Larger batch
	- EMA of weight 
	- Dropblock (structured dropout) @ FPN
	- IoU Loss in separate branch
	- IoU Aware: IoU guided NMS
	- Grid sensitive: introduced by [YOLOv4](yolov4.md). This helps the prediction after sigmoid to get to 0 or 1 position exactly, at grid boundary.
	- [CoordConv](coord_conv.md)
	- Matrix-NMS proposed by [SOLOv2](solov2.md)
	- SPP: efficiently boosts receptive field. 

#### Technical details
- Summary of technical details

#### Notes
- See this [review](https://mp.weixin.qq.com/s/pHOFqFihkkRVTYbkSTlG4w)

