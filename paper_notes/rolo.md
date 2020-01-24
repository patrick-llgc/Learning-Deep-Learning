# [ROLO: Spatially Supervised Recurrent Convolutional Neural Networks for Visual Object Tracking](https://arxiv.org/abs/1607.05781)

_January 2020_

tl;dr: Summary of the main idea.

#### Overall impression
There is a series of paper on recurrent single stage method for object detection. The main idea is to add RNN layer directly on top of the entire image feature. 

- [Recurrent YOLO](https://arxiv.org/abs/1607.05781)
- [Recurrent SSD](https://www.merl.com/publications/docs/TR2018-137.pdf)
- [Recurrent RetinaNet](https://doi.org/10.1007/978-3-030-04212-7_44)
- [Qualcomm Deep Radar Detector](http://openaccess.thecvf.com/content_ICCVW_2019/papers/CVRSUAD/Major_Vehicle_Detection_With_Automotive_Radar_Using_Deep_Learning_on_Range-Azimuth-Doppler_ICCVW_2019_paper.pdf)

The conversion of bbox to heatmap is also another example of transforming unstructured information to pseudo-image.

K = 6 frames

#### Key ideas
- Using historical visual semantics to improve tracking. (Although there is only one object in the image)
- When assigning detection to tracklet, use IoU distance between the current detection and the mean of its short-term history of validated detections.
- Training is multi-staged. First train the network on single image detection.
- Three inputs to LSTM: 
	- 4096-d feature vector for the entire image
	- heatmap from detection of the current frame
	- Output from last time-step

#### Technical details
- Evaluation metrics of tracking: Success Plots, accuracy (success ratio) vs IoU thresholds
	- OPE (one pass evaluation): all frames
	- TRE (temporal robustness evaluation): random frame as starting frame
	- SRE (spatial robustness evaluation): jittered GT bbox
- Under the same frames of GT, more video with sparse annotation is more useful than fewer video with dense annotation.
- YOLO with Kalman filter ([SORT](sort.md)) performs poorly due to fast motions, occlusions, and therefore occasionally poor detections.

#### Notes
- [github repo](https://github.com/Guanghan/ROLO)

