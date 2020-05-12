# [DF-VO: Visual Odometry Revisited: What Should Be Learnt?](https://arxiv.org/abs/1909.09803) [Depth and Flow for VO]

_May 2020_

tl;dr: Solving relative pose change with optical flow for 2D-2D matching is better than using unreliable depth prediction. Better than ORB-SLAM2 in some metrics. 

#### Overall impression
Solving for relative pose change between frames have two methods: 2D-2D matching and solve for essential matrix; lift 2D to 3D with predicted dense depth and 2D-3D matching and PnP.

Monocular VO suffer from scale-drift issue, thus VIO. This paper builds on [SC-sfm-learner](sc_sfm_learner.md), which also uses a geometric loss to ensure depth consistency and thus scale consistency.

DL based methods enable camera tracking in challenging conditions but they are not reliable and accurate in favorable conditions where geometry based algorithms is better (such as sufficient illumination and texture, sufficient overlap between frames). 

All learning based methods after [SfM-learner](sfm_learner.md) don't explicitly account for the multiview geometry constraints during inference. **Hybrid** methods achieves SOTA, such as [DF-VO](df_vo.md) and [D3VO](d3vo.md).

#### Key ideas
- Optical flow: LiteFlowNet
- Depth: SC-SfM-learner.
- Solving essential matrix for camera pose has limitations:
	- Scale ambiguity
	- Pure rotation issue and unstable solution under small translation
- Algorithm framework:
	- If optical flow >= threshold:
		- Use dense optical flow and 2D-2D matching to solve for essential matrix. Check cheirality.
		- Pick points with good bi-directional consistency (forward-backward)
	- Else:
		- Use depth prediction and solve 2D-3D matching with PnP. 
- The results outperforms purely DL based methods by a large margin, and even **beat ORB-SLAM2 consistently in relative pose estimation metrics RPE (relative pose error)**.
- Image resolution: at test time, simply increasing the image size to get more accurate correspondence which helps with relative pose estimation.

#### Technical details
- Resolving **scale drift** usually relies on 
	- Keeping a scale consistent map for map-to-frame tracking
	- global bundle adjustment for scale optimization
	- additional prior info such as constant camera height from known ground plan
	- leveraging IMU or speedometer ([packnet](packnet.md))
	- introduce geometric loss ([sc-sfm-learner](sc_sfm_learner.md)).
- ORB-SLAM2 occasionally suffers from tracking failure or unsuccessful initialization. Thus needs to run 3 times and pick the best one. 
- 36000 training pairs.

#### Notes
- [Demo video](https://www.youtube.com/watch?v=Nl8mFU4SJKY)
- [Github code](https://github.com/Huangying-Zhan/DF-VO)
