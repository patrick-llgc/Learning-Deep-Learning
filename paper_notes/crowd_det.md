# [CrowdDet: Detection in Crowded Scenes: One Proposal, Multiple Predictions](https://arxiv.org/abs/2003.09163)

_October 2020_

tl;dr: Multiple detections per anchor for crowd detection.

#### Overall impression
The paper proposed the idea of multiple instance prediction, and used EMD (earth mover distance) and set NMS to accommodate the multiple prediction per anchor. 

It achieves nearly 5% AP gain in [CrowdHuman](crowdhuman.md) dataset. 

Current works are either too complex or less effective for handling highly overlapped cases, or degrading the performance of less-overlapped cases.

![](https://github.com/Purkialo/images/blob/master/CrowdDet_arch.jpg?raw=true)

#### Key ideas
- Multiple instance prediction: The prediction of nearby proposals are expected to infer the **same set of instances**, rather than distinguishing individuals. 
	- Some cases are inherently difficult and ambiguous to detect and differentiate such as ◩ or ◪. 
	- this also greatly eases the learning in crowded scene. 
	- Each anchor predicts K (K=2) bboxes. When K=1, CrowdDet reduces to normal object detection.
- EMD (earth mover's distance) loss
	- For all permutaions of matching, select the best matching one with smallest loss
	- Add **dummy** boxes whose class label is regarded as Bg and without regression loss --> similar to the null padding in [DETR](detr.md).
- Set NMS: Do not suppress if the prediction are coming from the same proposal. 
- Refinement module
	- Concat predictions with feature vector and predict again. 
	- This is very similar to [IterDet](https://arxiv.org/abs/2005.05708).

#### Technical details
- Test on COCO to verify that there is no performance degradation rather than significant performance improvement. 
- [AP vs MR](ap_mr.md) in object detection.
	- AP is more sensitive to recall. MR is very sensitive to FP with high confidence. 

#### Notes
- [Pytorch code on Github](https://github.com/Purkialo/CrowdDet)

