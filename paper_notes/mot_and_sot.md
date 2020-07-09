# [MOT and SOT]

_July 2020_


#### SOT
- The aim of single object tracking (SOT) is to locate an object in continuous video frames given an initial annotation in the first frame. In contrast with MOT, there is only one object of interest and it is always in the considered image scene. Primarily focuses on designing sophisticated appearance models and/or motion models to deal with challenging factors such as scale changes, out of-plane rotations and illumination variations.
- SOT primarily focuses on designing sophisticated appearance models and/or motion models to deal with challenging factors such as scale changes, out-of-plane rotations and illumination variations

#### MOT
MOT task is a mid-level task in CV and is the foundation for high-level task such as pose estimation, action recognition and behavior analysis.

- In the MOT context, multiple objects with similar appearances and geometries in the searching area may confuse the single object tracker. 
- MOT additionally requires two tasks to be solved: determining the number of objects, which typically varies over time, and maintaining their identities.
- MOT is complicated by other factors:
	- 1) frequent occlusions
	- 2) initialization and termination of tracks
	- 3) similar appearance
	- 4) interactions among multiple objects

#### MOT classes
- Initialization methods
	- Detection-based tracking (tracking by detection)
	- Detection free tracking: manual initialization of a fixed number of objects in the first frames then localize them in subsequent frames. It cannot handle new object appearance.

#### Motion model
Why is it used? It estimates the potential position of objects in the future frames, thereby reducing the search space.

- Linear motion model: constant velocity
- 

#### Key ideas
- Summaries of the key ideas

#### Reference
- [Multiple Object Tracking: A Literature Review](https://arxiv.org/abs/1409.7618) [Mainly on pedestrian]
- [多目标跟踪任务介绍与评价规则](https://zhuanlan.zhihu.com/p/109764650)
- [从UMA Tracker(CVPR2020)出发谈谈SOT类MOT算法](https://zhuanlan.zhihu.com/p/138443415)
- [从CenterTrack出发谈谈联合检测和跟踪的MOT框架（含MOT17 No.1等多个榜前算法 ）](https://zhuanlan.zhihu.com/p/125395219)