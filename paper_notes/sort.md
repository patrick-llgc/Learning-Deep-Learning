# [sort: Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)

_September 2019_

tl;dr: detection results (bbox) tracking with kalman filter and Hungarian algorithm.

#### Overall impression
Classical methods for tracking reaches SOTA with faster RCNN. The performance gain is mainly from improved detection results.

The code is at [this github repo](https://github.com/abewley/sort). 

#### Key ideas
- Only bbox info is used for tracking. No appearance features are included. (the appearance features are included in the [deep-sort ](deep_sort.md) paper). 
- Estimation model: Kalman filter with linear constant velocity model between frames, independent of other objects and camera motion. When a detection is associated with a target, the detected bbox is used to update the target state (velocity). If no detection, then the state is simply the predicted without correction.
- Data association: Hungarian algorithm with IOU thresh.
- Creation and deletion of tracked objects: any detection with overlap less than IOU thresh. Tracks are are terminated if they are not detected for 1 frame. 

#### Technical details
- Aspect ratio is considered constant. 
- There are two schools of detection/tracking, detection by tracking and tacking by detection. SORT advocates tracking by detection to leverage the recent progress in DL in object detection.
- Two types of tracking
	- in batch mode, where future data is also available during tracking. 
	- online mode, where only history data is available. 

#### Notes
- Questions and notes on how to improve/revise the current work  

