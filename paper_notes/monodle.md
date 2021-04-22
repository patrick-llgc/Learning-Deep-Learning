# [monoDLE: Delving into Localization Errors for Monocular 3D Object Detection](https://arxiv.org/abs/2103.16237)

_April 2021_

tl;dr: Boost the performance of accurate 3D localization by using 2D bbox prediction task as auxiliary task, and discarding bad samples. 

#### Overall impression
This paper has a very thorough analysis into the error of mono3D. This reminds me of the [What monodepth see](what_monodepth_see.md) <kbd>ICCV 2019</kbd> paper. It founds that localization error is one vital factor accounting for the poor performance of mono3D. 

In addition, accurately localizing distant objects with existing technologies is almost impossible. So removing those distant objects can boost performance. 

The 3D location of an object can be decoupled into two parts, projected center of the 3D objects, and the depth.

![](https://cdn-images-1.medium.com/max/1600/1*2u0VtKiyeBSZpN2fSgLBPw.png)

#### Key ideas
- Misalignment of 2D and 3D centers
	- 2D center as representative point (object center). [CenterNet](centernet.md), [MonoPair](monopair.md) and [monoDIS](monodis.md), and both [MonoPair](monopair.md) and [monoDIS](monodis.md) also regresses an offset to estimate 3D center. 
	- 3D center as representative point. [SMOKE](smoke.md) regresses the 3D center directly and removed the 2D branches. 
- 3D center prediction
	- [MonoDLE](monodle.md) found that using 3D center can improve localization accuracy, and 2D detection is necessary as it helps to learn shared features for 3D detection.
	- 2D center is predicted via 3D center and an offset --> this is one key factor to improve performance.
	- Width and height of 2d bbox (generated from 3D bbox) are also predicted. Directly generating 2D bbox from 3D bbox yields suboptimal performance. --> Maybe we need 2D and 3D consistency loss.
- Losses
	- Depth prediction with aleatoric uncertainty
	- Yaw prediction with multibin loss and 12 bins.
- Hard negative mining is not always helpful. Clear domain gap between easy and hard cases exist in mono3D. Forcing network to learn from bad samples will reduce its representative ability for the others. 
	- Hard coding (discard distant cases) and soft coding (discount distant cases) works equally well
	- Discarding objects beyond 40 meters would actually decrease performance. 
- IoU oriented loss for 3D size estimation. Better than [monoDIS](monodis.md). 


#### Technical details
- Objects beyond 60 meters are removed from training. How do we predict 3D location of those objects when we have 8 MP cameras? --> Detect 3D location in the nominal focal distance and then scale with focal length. This scaling trick can actually be used in data augmentation. 
- It is amazing that the network can perform well with only ~3K training data. The entire KITTI dataset is only 12K. 
- Data augmentation: horizontal flipping, and random cropping/scaling for 2D tasks only.
- Confidence threshold: 0.2.
- Many mono3D methods use [DORN](dorn.md) as depth estimator, but this comparison is unfair as the training set for DORN overlaps with the valid set for mono3D.

#### Notes
- [code on github](https://github.com/xinzhuma/monodle)

