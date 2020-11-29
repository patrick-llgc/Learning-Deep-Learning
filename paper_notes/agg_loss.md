# [AggLoss: Occlusion-aware R-CNN: Detecting Pedestrians in a Crowd](https://arxiv.org/abs/1807.08407)

_October 2020_

tl;dr: Encourage different anchors to generate the same prediction, and occlusion-aware RoI pooling.

#### Overall impression
There are two main contributions of the paper. The first is AggLoss which encourages diff anchors associated with the same GT to output the same output. The second is the occlusion aware part-based RoI pooling.

Both [RepLoss](rep_loss.md) and [AggLoss](agg_loss.md) proposes additional penalties to produce more **compact** bounding boxes and become less sensitive to NMS. And also imposes additional penalties to bbox which appear in the middle of the two pedestrians. 

#### Key ideas
- **AggLoss**
	- If one GT bbox is associated with more than one anchors, encourages the prediction from all these anchors to be the same. It enforces SL1 loss between the avg prediction of the anchors and the corresponding GT. --> There seems to be something wrong in the paper's formulation. Shouldn't this be taking the avg of the abs (~SL1 loss) of the diff, rather than taking abs of the avg diff?
- **PORoI** (Part occlusion aware RoI pooling)
	- A part based model: inductive bias to introduce prior structure information of human body with visible prediction into the network.
	- The human body is divided into 5 parts, and each region U is compared with the visible region of bbox (V) to find IoU (intersection over U) to generate a binary visibility score. 
	- Additional "occlusion" loss is calculated with BCE loss.
	- The predicted visibility is used to modulate the pooled features before aggregating them with element-wise sum.

#### Technical details
- A common trick to boost pedestrian detection performance: x1.3 resolution. 

#### Notes
- Questions and notes on how to improve/revise the current work  

