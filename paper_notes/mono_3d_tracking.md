# [Joint Monocular 3D Vehicle Detection and Tracking](https://arxiv.org/abs/1811.10742) [[Notes](paper_notes/mono_3d_tracking.md)]

_October 2019_

tl;dr: Add 3D tracking with LSTM based on mono3d object detection.

#### Overall impression
An autonomous driving system has to estimate: 1) 2d detection, 2) distance, orientation, 3d size 3) association and 4) prediction of future movement.

The authors rely on mono 3DOD and perform tracking on it. The estimation of 3d bbox largely extends [deep3dbox](deep3dbox.md). It also directly regresses inverse distance (1/d, or disparity), and is trained with roiAligned features instead of using image patches such as deep3Dbox.

#### Key ideas
- Regress 2D projection of 3D center along side RPN. --> 2D center wrongly locates the 3D center. Amodal object detection with amodal annotation handles occlusion better, but not for truncated cases.
- With RoIAligned features, distance (disparity, 1/d), orientation, 3D size are estimated. 
- Solve data hungry problem: use synthetic data. The paper extends FSV.

#### Technical details
- Human stereo vision only reaches several meters.
- Tracking is increasingly difficult especially in the presence of large ego motion (a turning car).
- Intrinsics and extrinsics calculated from GPS or IMU. 
- The regression of distance is on the L1 loss on depth, and the reprojected 3D center. (We know the GT 3D bbox center)
- Depth ordering of tracks before matching using Hungarian (Kuhn-Munres) algorithm
- occlusion aware tracking: added occluded status in addition to the normal {birth, tracked, lost, death} set. If an object is occluded more than 70%, then its feature representation and lifespan is not updated. 

#### Notes
- The authors have a [youtube video](https://www.youtube.com/watch?v=EJAtOCKI31g) for this paper.
- During association between radar/camera, maybe we should use depth ordering with ymin.

