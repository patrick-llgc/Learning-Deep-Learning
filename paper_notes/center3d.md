# [Center3D: Center-based Monocular 3D Object Detection with Joint Depth Understanding](https://arxiv.org/abs/2005.13423)

_November 2020_

tl;dr: CenterNet-based approach, with better distance estimation

#### Overall impression
The paper proposed two approaches for distance estimation. One is based on DORN with better discretization strategy, and the second is based on breaking down the distance into two large bins, one for near objects and the other for faraway ones. 

Overall this paper is a very solid contribution to monocular 3D object detection. Nothing fancy, but concrete experiment and small design tweaks. 

A quick summary of [CenterNet](centernet.md) monocular 3D object detection.

- [CenterNet](centernet.md) predicts 2D bbox center and uses it as 3D bbox center. 
- [SMOKE](smoke.md) predicts projected 3D bbox center.
- [KM3D-Net](km3d_net.md) and [Center3D](center3d.md) predict 2D bbox center and offset from projected 3D bbox center. 

#### Key ideas
- 2D and projected 3D center are different
	- the gap decreases for faraway objects and which appear in the center area of the image plane.
	- The gap becomes significant for objects that are close to the camera or on the image boundary.
- LID (linear increasing discretization)
	- The SID (space-increasing discretization) approach used by [DORN](dorn.md) gives too dense bins in the unnecessary nearby range. 
	- The length of the bins increases linearly in LID (and log-wise in SID).
	- [DORN](dorn.md) counts the number of bins with proba > 0.5 as ordinal label and use the median value of that bins the estimated depth in meters. 
	- LID also uses a regression bit to predict the residual value. --> This is very important to ensure good depth estimation as shown in the ablation study.
- DepJoint: piece wise depth prediction
	- Breaking the distance into two bins (either overlapping or back-to-back bins)
	- Eigen's exponential transformation of distance: $\Phi (d) = e ^ {-d}$.
	- This has very good accuracy in close range, but not so in distance range
	- Augment the prediction for faraway objects by also predicting $d' = d_{max} - d$. Then during inference, uses the weighted prediction of the two prediction.
	- The bin breakdown is controlled by two hyper parameters. The bins can have overlap or back-to-back.

#### Technical details
- RA (reference area) solves the issue of lack of supervision for attribute prediction. Not only the GT center point contribute to the attribute prediction losses, but a dilated support region is used to predict all the attribute. --> this is inspired by the support region in [SS3D](ss3d.md).

#### Notes
- Questions and notes on how to improve/revise the current work  

