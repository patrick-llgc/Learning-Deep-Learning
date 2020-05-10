# [Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video](https://arxiv.org/abs/1908.10553)

_September 2019_

tl;dr: First paper that demonstrate scale consistency in long video and can achieve better performance than stereo. 

The next step paper is [DF-VO](df_vo.md) which predicts dense optical flow and uses 2D-2D matching to regress ego-motion, achieving even more accurate VO. 

#### Overall impression
The introduction of depth scale consistency is the key to the good performance on relative pose estimation, and thus enables the VO use. 

The performance of [sfm-learner](sfm_learner.md) is actually not that good on VO. Scale and rotation drift is large. See [scale consistent sfm-learner](sc_sfm_learner.md) for better VO performance.
![](https://pic2.zhimg.com/80/v2-7425fdf248804f0c900d455ab0de5d51_1440w.jpg)

#### Key ideas
- The main idea is simple: to ensure that the depth is consistent across frames. The consistency in depth will lead to scale consistency.

#### Technical details
- Summary of technical details

#### Notes
- code on [github](https://github.com/JiawangBian/SC-SfMLearner-Release).
- Review on [知乎](https://zhuanlan.zhihu.com/p/83901104)