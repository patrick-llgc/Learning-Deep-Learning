# [CBGS: Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection](https://arxiv.org/abs/1908.09492)

_July 2020_

tl;dr: Class rebalance of minority helps in object detection for nuscenes dataset. 

#### Overall impression
The class balanced sampling and class-grouped heads are useful to handle imbalanced object detection. 

#### Key ideas
- **DS sampling**: 
	- increases sample density of rare classes to avoid gradient vanishing
	- count instances and samples (frames). Resample so that samples for each class is on the same order of magnitude.
- **Class balanced grouping**: each group has a separate head.
	- Classes of similar shapes or sizes should be grouped. 
	- Instance numbers of diff groups should be balanced properly.
	- Supergroups:
		- cars (majority classes)
		- truck, construction vehicle
		- bus, trailer
		- barrier
		- motorcycle, bicycle
		- pedestrian, traffic cone
- Fit ground plane and plant GT back in.
- Bag of tricks
	- Accumulate 10 frames (0.5 seconds) to form a dense lidar BEV
	- AdamW + 1 cycle LR

#### Technical details
- Regress vx and vy. If bicycle speed is above a certain thresh, then it is with rider. 

#### Notes
- Questions and notes on how to improve/revise the current work  

