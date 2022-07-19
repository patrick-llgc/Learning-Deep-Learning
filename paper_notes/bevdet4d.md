# [BEVDet4D: Exploit Temporal Cues in Multi-camera 3D Object Detection](https://arxiv.org/abs/2203.17054)

_June 2022_

tl;dr: Lift BEVDet to 4D with limited temporal information. 

#### Overall impression
The paper builds on top of [BEVDet](bevdet.md) and introduces temporal component. This boosts the accuracy of velocity estimation and thus NDS. 

The spatiotemporal alignment module is very similar to that from [BEVFormer](bevformer.md), yet with a much simpler fusion module. The temporal fusion is not through convLSTM or similar recurrent structure but simply concats with previous frame T-1. The time window is not adjustable. This could limit the performance of BEVDet. 

The engineering work is still excellent, but the writing unfortunately lacks clarity and needs some guess work. The math equations in this paper are really unnecessary.

This work can be compared with the concurrent [BEVerse](beverse.md). In comparison, BEVDet4D has slightly better performance in BEV detection.

#### Key ideas
- Naively concat BEV feature leads to performance degradation. This shows that spatiotemporal alignment is necessary to ensure good BEV performance. 
- Spatiotemporal alignment simplifies the task of velocity estimation 
	- it removes the need to learn position shifting relevant to ego motion. 
	- Also after the alignment the velocity estimation and orientation is strongly correlated.
- Time window is only 2. May be improved if longer time window is used. --> Similar to [BEVDepth](bevdepth.md).

#### Technical details
- [PETR](petr.md) uses heavy data augmentation, like [BEVDet](bevdet.md). The data augmentation in image view (IV) is a bit puzzling but could lead to more robust behavior during inference. 
- mAVE metric as a widely accepted metric for velocity estimation. 
- In the first version of the paper, the author claimed that changing velocity estimation to translation prediction alone helps cut the velocity estimation error by half. --> This is actually a bug and has been fixed in the new version of the manuscript. The two should be equivalent, given constant interframe time. However, predicting offset may be a good strategy to do temporal data aug. But then we could simply scale the velocity GT.


#### Notes
- Code on [Github](https://github.com/HuangJunJie2017/BEVDet)
