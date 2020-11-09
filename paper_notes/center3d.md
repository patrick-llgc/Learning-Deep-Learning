# [Center3D: Center-based Monocular 3D Object Detection with Joint Depth Understanding](https://arxiv.org/abs/2005.13423)

_November 2020_

tl;dr: CenterNet-based approach, with better distance estimation

#### Overall impression
The paper proposed two approaches for distance estimation. One is based on DORN with better discretization strategy, and the second is based on breaking down the distance into two large bins, one for near objects and the other for faraway ones. 

It is [CenterNet](centernet.md) based approach, very similar to [SMOKE](smoke.md) and [KM3D-Net](km3d_net.md).

#### Key ideas
- LID (linear increasing discretization)
	- The SID (space-increasing discretization) approach used by [DORN](dorn.md) gives too many bins in the nearby range. 
- DepJoint
	- Breaking the distance into two bins (either overlapping or back-to-back bins)

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

