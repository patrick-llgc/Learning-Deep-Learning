# [Adaptive NMS: Refining Pedestrian Detection in a Crowd](https://arxiv.org/abs/1904.03629)

_October 2020_

tl;dr: Predict the object density (crowdedness) score and use it as adaptive threshold for NMS.

#### Overall impression
The paper has a simple intuition: that high NMS threshold keeps more crowded instances while a low NMS threshold wipes out more false positives. The key question is how to predict the crowdedness in inference time? The paper proposes to predict it together with other attributes of bboxes. 

Both [RepLoss](rep_loss.md) and [Occlusion aware R-CNN](orcnn.md) proposes additional penaltiies to produce more compact bounding boxes and become less sensitive to NMS. 

#### Key ideas
- Basic hypothesis under NMS and soft-NMS: detection bboxes with higher overlap with bbox of interest M should have a higher likelihood of being FP. This assumption does not hold true in crowded regions. If M is in a crowded region, its highly overlapped neighboring proposals are likely to be TP.
- For object in high object density area, use the max(thresh, object_density) to perform NMS. This adaptively adjust up the threshold in crowded regions. 
- Density prediction subnet:
	- Object density: max bbox IoU with other objects in the GT set. It is a score between 0 to 1, and naturally can be used to adjust the NMS threshold. 
	- On top of RPN for two stage detectors, taking the objectness predictions, bounding box predictions and conv features as input. 

#### Technical details
- [AP vs MR](ap_mr.md) in object detection.
	- Combining adaptive NMS and soft-NMS has minor or even negative improvements on metric MR^-2 (0.01 to 1 FPPI). Reason may be the benefit happens beyond 1 FPPI and thus does not improve metric. 
- Reasonable: Bare (0 to 0.1), Partial (0.1 to 0.35), Heavy (0.35 to 1).

#### Notes
- Questions and notes on how to improve/revise the current work  

