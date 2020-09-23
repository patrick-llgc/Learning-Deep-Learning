# [InstanceMotSeg: Real-time Instance Motion Segmentation for Autonomous Driving](https://arxiv.org/abs/2008.07008)

_September 2020_

tl;dr: Introduce the concept of instance into motion segmentation.

#### Overall impression
Motion segmentation will enable the detection of objects unseen during training (rare animals, or rare type of trucks), and thus is a crucial task.

Semantic segmentation and motion segmentation usually do not have the concept of instances, but focus on pixel level performance.

#### Key ideas
- Architecture based on [YOLACT](yolact.md).
	- Two streams of inputs: RGB and OF (optical flow) --> better than take in two consecutive RGB frames 
	- feature fusion in FPN
	- generate shared prototype masks for instance semantic segmentation and instance motion segmentation
	- two shallow prediction heads predict diff coefficients for instance semantic segmentation and instance motion segmentation
- The feature extraction network, feature pyramid network and prototype masks are **shared** among two tasks, but the **learned coefficients** to construct masks and learned bbox regression are different. 


#### Technical details
- Static vehicles naturally occur in prototype masks, and coefficients of semantic instance seg and motion instance seg have different coefficients for that prototype mask. This enables the network to capture semantic segmentation for that static car, but ignore it in the motion instance seg results. 
- Corner cases:
	- When objects are moving parallel to the ego vehicle, then it is much more difficult to do motion segmentation.
	- VOS (video object segmentation) setting is not best suited for autonomous driving as moving objects at larger distance don't appear to be salient.
- Geometry based motion analysis do not exploit global context.
- Nviaid Xavier has a hardware accelerator for computing dense optical flow and can be leveraged without requiring additional processing. --> what is it?

#### Notes
- [Demo on youtube](https://www.youtube.com/watch?v=CWGZibugD9g)

