# AP vs MR vs F1 in object detection

_October 2020_

tl;dr: Summarizes the different metrics used in general object detection vs pedestrian detection.

#### Overall impression
Both AP and MR are used heavily in academia in evaluating object detection algorithms, just like AUC of ROC for classifier. But in deployment, neither MR or AP is a good metric, as we have to select one working point. The good old F1 score may be better.

**In other words, mAP is used to evaluate detection algorithms, and acc (or F1 score) is used to evaluate detectors in specific scenarios. The former is used in academia and papers, and the latter is used in industry and production.**


#### Key ideas
- mAP (mean average precision), average over diff thresh and categories.
- mMR (average log miss rate over FP per image)
- Log average miss rate on False Positive Per Image (MR^-2) is usually the KPI for pedestrian detection. This looks like FROC curve. Miss rate = 1 - recall. MR score is plot on both logx and logy. The lower the better. 

- 行人检测和通用目标检测的区别，主要有两点：(1). 检测目标类别数不同; (2). 评测指标不同。第一点大家很容易理解，主要谈谈第二点。通用目标检测的评测指标是mAP@0.5-0.95（越高越好），而行人检测的评测指标是mMR (Log-average Miss Rate)（越低越好）。mAP是对Precision和Recall做整体评估，即P-R曲线下的面积，在这个指标下的低分TP可以带来Recall的提升，因此mAP指标也会提升，这也是RetinaNet[3]涨点的一方面，如果观察其P-R曲线就可以发现很长的尾巴。而mMR则是在FPPI@0.01-1（平均每张图FP数）下Miss（漏检）的平均，很明显这个指标同时关注FP（误检）和FN（漏检），因此mAP高的模型不一定mMR低。以上就是两个评测指标的区别，两个各有优劣，适用场景不同，例如对于行人检测来说，更重要的是减少FP，如果能减少高分FP将在指标上带来很大的提升。 -- [Review on 知乎](https://zhuanlan.zhihu.com/p/95253096)
- AP is more sensitive to recall. MR is very sensitive to FP with high confidence. 
- MR only cares about the predicted bboxes whose scores are higher than the highest scored FP

#### SoftNMS
- Combining adaptive NMS and soft-NMS has minor or even negative improvements on metric MR^-2 (0.01 to 1 FPPI). Reason may be the benefit happens beyond 1 FPPI and thus does not improve metric. [Adaptive NMS](adaptive_nms.md)
- Soft-NMS maintains lots of long-tail detection results for improving recall at the expense of bringing more false positives, which leqds to negative impact on human detection especially for the metric of MR (where FP with high score is the bottleneck).

#### Papers on pedestrian detection (in a crowd)
- [RepLoss: Repulsion Loss: Detecting Pedestrians in a Crowd](https://arxiv.org/abs/1711.07752) [[Notes](rep_loss.md)] <kbd>CVPR 2018</kbd> [crowd detection, Megvii]
- [Adaptive NMS: Refining Pedestrian Detection in a Crowd](https://arxiv.org/abs/1904.03629) [[Notes](adaptive_nms.md)] <kbd>CVPR 2019 oral</kbd> [crowd detection, NMS]
- [Occlusion-aware R-CNN: Detecting Pedestrians in a Crowd](https://arxiv.org/abs/1807.08407) [[Notes](orcnn.md)] <kbd>ECCV 2018</kbd> [crowd detection]
- [CrowdDet: Detection in Crowded Scenes: One Proposal, Multiple Predictions](https://arxiv.org/abs/2003.09163) [[Notes](./crowd_det.md)] <kbd>CVPR 2020 oral</kbd> [crowd detection, Megvii]
- [Double Anchor R-CNN for Human Detection in a Crowd](https://arxiv.org/abs/1909.09998) [[Notes](./double_anchor.md)] [head-body bundle]


#### Notes
- [浅析经典目标检测评价指标--mmAP（一）](https://zhuanlan.zhihu.com/p/55575423)
- [浅析经典目标检测评价指标--mmAP（二）](https://zhuanlan.zhihu.com/p/56899189)

