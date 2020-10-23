# [Double Anchor R-CNN for Human Detection in a Crowd](https://arxiv.org/abs/1909.09998)

_October 2020_

tl;dr: Double Anchor RPN is developed to capture body and head parts in pairs.

#### Overall impression
Crowd occlusion is challenging for two reasons:

- when people overlap largely with each other, semantic features of different instances also interweave and make sectors difficult to discriminate instance boundaries. 
- Even though detectors successfully differentiate and detect instances, they may be suppressed by NMS.

The intuition behind the paper is simple: compared with the human body, the head usually has a smaller scale, less overlap and a better view in real-world images, and thus is more robust to pose variations and crowd occlusions. --> this has very similar motivation to [R2 NMS](r2_nms.md) and [VG NMS](vg_nms.md).

One main challenge in crowd detection is high score false positives. --> However safety-wise this does not seem to be an issue for autonomous driving. 

#### Key ideas
- **Double Anchor RPN** basically is to output two regressed offsets (for body and head) and one score.
- Proposal Crossover: 
	- Two branches: head-body branch which regresses head and body from head anchor, and body-head branch which regresses head and body from body anchor
	- Body proposals from head-body branch is not good. Thus perform IoU check of the body proposals between head-body branch and body-head branch, and replace body proposal from head-body branch (lower quality) with that from the body-head branch (higher quality)
- Feature aggregation:
	- perform RoIAlign on two proposals separately, then concat
	- predict head bbox loc/score, and body bbox loc/score.
	- this is further enhanced by the PPFE (paired proposal feature extractor) module in [R2 NMS](r2_nms.md).
- Joint NMS: 
	- weighted score from both head bbox score and body bbox score. 
	- If head IoU or body IoU exceeds certain threshold then suppress

#### Technical details
- [AP vs MR](ap_mr.md) in object detection.
	- Soft-NMS maintains lots of long-tail detection results for improving recall at the expense of bringing more false positives, which leqds to negative impact on human detection especially for the metric of MR (where FP with high score is the bottleneck).
	- Note that in deployment, neither MR or AP is a good metric, as we have to select one working point. 

#### Notes
- [Review on Zhihu](https://zhuanlan.zhihu.com/p/95253096)

