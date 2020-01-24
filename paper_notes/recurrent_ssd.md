# [Recurrent SSD: Recurrent Multi-frame Single Shot Detector for Video Object Detection](https://www.merl.com/publications/docs/TR2018-137.pdf)

_January 2020_

tl;dr: Using history to boost object detection on KITTI.

#### Overall impression
There is a series of paper on recurrent single stage method for object detection. The main idea is to add RNN layer directly on top of the entire image feature. 

- [Recurrent YOLO](https://arxiv.org/abs/1607.05781)
- [Recurrent SSD](https://www.merl.com/publications/docs/TR2018-137.pdf)
- [Recurrent RetinaNet](https://doi.org/10.1007/978-3-030-04212-7_44)
- [Qualcomm Deep Radar Detector](http://openaccess.thecvf.com/content_ICCVW_2019/papers/CVRSUAD/Major_Vehicle_Detection_With_Automotive_Radar_Using_Deep_Learning_on_Range-Azimuth-Doppler_ICCVW_2019_paper.pdf)

Another way to look at feature aggregation over time is data fusion. Instead of fusing information from different sensors, it is fusing information from different time-stamp. The fusion technique can be element wise (addition or max), concatenation or recurrent layer. 

This is perhaps the best clean solution to video object detection problem. Much cleaner than [ROLO](rolo.md).

K=4 frames

#### Key ideas
- Augment SSD meta-architecture by conv-recurrent layer (conv-GRU). This maintains the fully-convolutional feature of SSD, keeping it fast.
- Two ways to integrate information from multiple frames. These two are orthogonal to each other and can be used together. 
	- Feature level: accumulate feature maps across time. Such as [towards high performance video object detection](high_performance_video_od.md).
	- Box level: tracking by detection.
- **It does not require extra labeled training data as only the final time-stamped image needs labeled bounding boxes.** 
- The aggregated feature map in the recurrent layer can be used for visualization. It can recover heavily occluded object (similar to [ROLO](rolo.md)). 


#### Technical details
- Late concatenation (right after backbone and before detection head), with additional conv layer achieves almost as good performance, but it is taking 4 images as input and slows down the system a lot. This can be improved by a circular buffer but adds to the complexity of the system.
- Recurrent SSD achieves over 2.7 mAP improvement over single frame SSD on KITTI.

#### Notes
- [Association LSTM](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lu__Online_Video_ICCV_2017_paper.pdf) is inspired by Siamese network for re-identification and works Hungarian algorithm to be jointly trained with neural network.
- How to efficiently introduce LSTM with >1 stride? Keeping two states?

