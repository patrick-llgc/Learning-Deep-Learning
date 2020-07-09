# [CenterTrack: Tracking Objects as Points](https://arxiv.org/abs/2004.01177)

_July 2020_

tl;dr: Use [CenterNet](centernet.md) to predict offset between neighboring frames. Nearest neighbor would work well 

#### Overall impression
CenterNet achieves the [SOTA of mono3D on nuScenes as of 07/04/2020](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera). On nuScenes it achieves 34 mAP, almost as good as LIDAR based approach one year ago by [PointPillars](pointpillars.md) at 31 mAP. (The lidar SOTA performance is refreshed by [CenterPoint](centerpoint.md) to 60 mAP.)


##### The complete cycle
In early days of computer vision, tracking was phrased as following **interest points** through space and time. It then got beaten by "tracking by detection" (or tracking following detection) which follows detected bounding box throughout time. The tracking stage is usually slow and complicated association strategy (Hungarian matching) by either IoU of 2D bbox or learnt embedding vectors of each object, or with motion model and EKF. The simple displacement prediction is akin to **sparse optical flow**. KLT (Kanade Lucas Tomasi) contains GFT (good feature to track) + LK (Lucas Kanade <金出武雄@CMU>). CenterNet is like GFT and the offset vector is like LK.

The first simultaneous detection and tracking paper [Tracktor](tracktor.md) which predicts bbox detection and bbox offset for tracking simultaneously. [CenterTrack](centertrack.md) simplifies the process further by reducing bounding box to a point with attributes. 

Previous joint detection and tracking work are all based on Faster RCNN two-stage framework where the tracked bbox are used as region proposals. This has two issues: 1) It assumes that the bounding box has large overlaps which is not true in low-frame rate regimes (such as in nuScenes) 2) it is hard to run MOT in real time. 

It feeds detection results from previous frame as additional input to boost performance in the current frame. This **tracking-conditioned detection** is like autoregressive models. This trick is also used in [ROLO](rolo.md). This provides a temporally coherent set of detection objects. (Only location, not size). 

#### Key ideas
- CenterTrack localizes objects and predicts their association with the previous frame. It can be used for 2D OD or 3D OD based on monocular camera images. The detector is trained to output an **offset vector** from the current object center to its center in previous frames. Then a greedy matching suffices for object association. 
- Input: Previous frame, previous center point frame heatmap, current frame. The prior detections are encoded as a class-agnostic single channel heatmap. The rendering is the same as detection GT encoding.
- CenterTrack is a **local** tracker, in that it identifies across consecutive frames without reestablishing association across temporal gaps. 
- Association with offset: greedily find object with the highest conf score within a radius $\kappa = \sqrt {wh}$. There is no need for complicated distance metric or graph matching.
- Data augmentation
	- Jitter location by 0.05 w or h multiplied by a Gaussian random variable
	- Randomly inject FP near GT object locations with probability $\lambda_{fp} = 0.1$
	- Simulate FN by randomly removing detection with probability $\lambda_{fn} = 0.4$
	- We can train centerTrack with static images by randomly scaling and shifting images
	- Without noise injection, MOTA drops dramatically from 66 to 34 points.
- Evaluation metric: AMOTA (averaged MOTA at different thresholds). Note that MOTA can be negative and thus usually capped at 0.
- Runtime: 82 ms on KITTI and 45 ms on nuScenes.

#### Technical details
- Center based object detection is easily extended from 2D to 3D as 2D bbox does not have to be first estimated. --> this is true for 3D anchored design such as [M3D-RPN](m3d_rpn.md).
- Data augmentation by random input dropout and noise injection to prevent neural network to cheat from the detection in previous frame. 
- It can be trained on labeled video sequences, or on static images with aggresive data augmentation. The static image training works better if the frame rate is high enough in reality.
- [CenterNet](centernet.md) uses additional 2 ch to correct quantization error. 
- [CenterTrack](centertrack.md) can be finetuned from [CenterNet](centernet.md).
- NuScenes dataset has video data at 12 FPS but only annotated at 2 FPS. The interframe motion is significant.
- MOT dataset works on amodal bbox. To handle **cropped objects**, the formulation is changed to that of [FCOS](fcos.md), with center point and four offset to four edges. That means object center may not be at the bbox center.
- Evaluation results
	- Tracking with public detection is better than other methods
		- Lower FN due to tracking conditioned pipeline
		- Simple learned offset is effective.
	- On nuScenes, the mono3D tracking is almost 3 times better as existing pipeline ([monoDIS](monodis.md) and [AB3D](ab3d.md)), from 7 AMOTA to 28 AMOTA.
		- EKF works not that well at low framerates on nuScenes. Removing heatmap does not hurt that much, but removing offset reduces AMOTA by 10 points. 
		- In MOT17 or KITTI, without predicting offset works pretty well. 
	- On KITTI, compare with other motion model, Kalman filter only works almost as well as no motion model. Optical flow with [flownet2](flownet2.md) (sampling at object center) works almost as well as centerTrack. (But flownet2 runs at 150 ms per image pair.)



#### Notes
- [Review on 知乎](https://zhuanlan.zhihu.com/p/125395219) for [CenterTrack](centertrack.md), [Detect to track and track to detect](detect_track.md) and [tracktor++](tracktor.md).
- [Github repo](https://github.com/xingyizhou/CenterTrack)
	- Follow [installation guide](https://github.com/xingyizhou/CenterTrack/blob/master/readme/INSTALL.md)
	- Need to downgrade pytorch to 1.2 to work with cuda 10.0
	- Need to downgrade sklearn `pip install scikit-learn==0.22.2`
- Next step: combine local tracking such as centerTrack with long term tracking methods such as [SORT](sort.md) with EKF.
