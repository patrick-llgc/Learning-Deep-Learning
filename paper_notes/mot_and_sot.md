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

#### IoU based tracker
- They do not have motion models.
- [IOU Tracker](http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf)
	- At high fps, just use IoU to associate tracks with bbox (You can actually write a paper about this simple idea?)
	- ![](https://pic4.zhimg.com/80/v2-1f662f054e51b6cf9da7cc0f13a05717_1440w.jpg)
- [Visual-IOU Tracker](http://elvera.nue.tu-berlin.de/files/1547Bochinski2018.pdf)
	- If cannot associate, then spawn SOT trackers at each failed bbox.
	- This extends the IoU tracker, just like deepSORT extends SORT.
	- ![](https://pic2.zhimg.com/80/v2-3c9e2cd09efce7bbd2684aa824c50645_1440w.jpg)


#### Key ideas
- 一般提到“视觉目标跟踪”或“VOT”，往往指的是单目标跟踪。尽管看起来SOT（Single Object Tracking）和MOT（Multi Object Tracking）只是目标数量上的差异，但它们通用的方法实际上截然不同。从研究对象上讲，单目标跟踪算法一般是不限类别的，而多目标跟踪一般是仅针对特定类别的物体。从时长上讲，单目标跟踪更多地针对短时间的图像序列，而多目标跟踪一般要处理较长的视频，其中涉及各个目标的出现、遮挡和离开等情况。从实现思路上讲，单目标跟踪更关注如何对目标进行重定位，而常见的多目标跟踪方法往往更多地关注如何根据已检测到的目标进行匹配。 [source](https://zhuanlan.zhihu.com/p/76153871)

#### Reference
- [Multiple Object Tracking: A Literature Review](https://arxiv.org/abs/1409.7618) [Mainly on pedestrian]
- [多目标跟踪任务介绍与评价规则](https://zhuanlan.zhihu.com/p/109764650)
- [从UMA Tracker(CVPR2020)出发谈谈SOT类MOT算法](https://zhuanlan.zhihu.com/p/138443415)
- [从CenterTrack出发谈谈联合检测和跟踪的MOT框架（含MOT17 No.1等多个榜前算法 ）](https://zhuanlan.zhihu.com/p/125395219)
- [一文带你了解视觉目标跟踪](https://zhuanlan.zhihu.com/p/76153871)