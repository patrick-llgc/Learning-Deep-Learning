# [NMS by Representative Region: Towards Crowded Pedestrian Detection by Proposal Pairing](https://arxiv.org/abs/2003.12729)

_October 2020_

tl;dr: Predict both full bbox and visible region and use visible region for NMS. 

#### Overall impression
This paper resembles that of [Visibility guided NMS](vg_nms.md), which focuses on crowd vehicle detection. The contribution of this paper is the introduction of the PPFE (paired proposal feature extractor) module. This takes the feature aggregation in [Double Anchor](double_anchor.md) to a new level.

#### Key ideas
- Visible parts of pedestrians by definition **suffer much less from occlusion**, a relative low IoU thresh sufficiently removes the redundant bboxes locating the same pedestrian, and meanwhile avoids the large number of FP. --> This has the same motivation as [Double Anchor](double_anchor.md).
	- The IoU between visible regions of two bboxes is a better indicator showing if two full body bboxes belong to the same pedestrian. 
- The visible bbox and full bbox have high overlaps and is not as different as head-body bboxes, and thus can more reliably get regressed from the same anchor, as compared to [Double Anchor](double_anchor.md).
- NPM (Native pair model) + PPFE (paired proposal feature extractor / feature aggregator) = PBM (paired box model).
	- The feature aggregation, even the simple concat config, boosts the paired box model hugely.
- With R2-NMS (NMS by visible region)
	- MR not too much change: MR only cares about the predicted bboxes whose scores are higher than the highest scored FP. --> See [AP vs MR](ap_mr.md)
	- AP boosts hugely


#### Technical details
- Inter-class occlusion is easier to solve than intra-class occlusion (crowd occlusion).
- In [CrowdHuman](crowdhuman.md) dataset, if set NMS IoU = 0.5 in GT, nearly 10% of the GT instanced will be suppressed. At IoU = 0.7, only 1% will be suppressed. 
- The GT assignment as compared with Faster RCNN is more restrictive. It requires both the full bbbox IoU with anchor and the visible bbox IoU with anchor to be above 0.7.
- CityPersons: reasonable occlusion < 0.35. Heavy occlusion [0.35, 0.8]. 

#### Notes
- [Summary by 1st author on Zhihu](https://zhuanlan.zhihu.com/p/68677880)
