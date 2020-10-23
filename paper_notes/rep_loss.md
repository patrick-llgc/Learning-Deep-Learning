# [Repulsion Loss: Detecting Pedestrians in a Crowd](https://arxiv.org/abs/1711.07752)

_October 2020_

tl;dr: A novel bbox regression loss specifically designed for crowd scenes. This not only push each proposal to reach its designed target, but also keep it away from other surrounding objects.

#### Overall impression
The paper has a solid analysis into the difficulty of detection under crowd occlusion. 

Two issues with crowd occlusion: 1) increases the difficulty of bbox localization, as it is hard to tell diff GT apart as regression target. 2) NMS is more sensitive to threshold (higher thresh brings more FP, and lower thresh leads to missed detections).

Thus the bbox regression with RepLoss is driven by two motivations: attraction by the target and repulsion by other surrounding target (GT) and proposals (pred).

Both [RepLoss](rep_loss.md) and [Occlusion aware R-CNN](orcnn.md) proposes additional penalties to produce more **compact** bounding boxes and become less sensitive to NMS. And also imposes additional penalties to bbox which appear in the middle of the two pedestrians. 

Visualization before NMS seems to be a powerful debugging tool.

#### Key ideas
- Occlusion: inter-class and intra-class. Intra-class occlusion is also named crowd occlusion, which happens when an object is occluded by objects of the same category.
- Repulsion terms: 
	- **RepGT**: Intersection over GT (to avoid prediction from cheating by increasing pred bbox) with smooth ln loss from [UnitBox](https://arxiv.org/abs/1608.01471). It penalizes overlap with non-target GT object. 
	- **RepBox**: the IoU region between two predicted bboxes with different designated targets needs to be small. This means the predicted bboes with diff regression targets are more likely to be merged into one after NMS. 
- The selection of IoU or IoG is due to their boundedness within [0, 1]. 
- Smooth ln loss: more robust to outliers. 
	- pred bboxes are much denser than the GT boxes, a pair of two pred bboxes are more likely to have a larger overlap than a pair of one predicted box and one GT box. Thus RepBox is more likely to have outliers than in RepGT.

#### Technical details
- [AP vs MR](ap_mr.md) in object detection.
	- Log average miss rate on False Positive Per Image (MR^-2) is usually the KPI for pedestrian detection. This looks like FROC curve. Miss rate = 1 - recall. MR score is plot on both logx and logy. The lower the better. 
- Occlusion: occ > 0.1. Occ is calculated by 1 - (visible bbox area / full bbox area). Crowd occlusion: occ > 0.1, IoU > 0.1
- Occlusion < 35%. [0, 10%]: bare, [10%, 35%] partial, [35%, 1): heavy. Bare and partial occlusions are **reasonable** occlusions.
- FP: background (0 GT under 0.1 IoU), localization error (1 GT), and crowd error (2+ GT).


#### Notes
- Questions and notes on how to improve/revise the current work  

