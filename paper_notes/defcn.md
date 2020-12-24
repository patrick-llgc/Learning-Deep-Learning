# [DeFCN: End-to-End Object Detection with Fully Convolutional Network](https://arxiv.org/abs/2012.03544)

_December 2020_

tl;dr: End to end object detection with good label assignment. 

#### Overall impression
The study build upon [FCOS](fcos.md). It pointed out that the one-to-many label assignment makes NMS necessary. Thus a good one-to-one policy is the key. A hand crafted one-to-one label assignment already yields OK-ish performance (10% relative drop in KPI). The paper is also inspired by MultiBox and [DETR](detr.md) to use bipartite matching as the matching cost, to allow neural network to learn a better assignment policy.

#### Key ideas
- One-to-one label assignment is key.
	- One-to-one based on center or anchor is already OK. 
	- Matching cost by foreground loss (as in [DETR](detr.md)) improves KPI
	- Modified POTO (prediction aware one to one) cost for matching is even better, as the foreground loss (cls+reg) may be weighted and it may not be optimal for bipartite matching.
	- The selection of matching cost is not necessarily differentiable. So theoretically we can use mAP as the cost --> see [review on Zhihu](https://mp.weixin.qq.com/s/lfkcbfQrFOGUpkGB7Oh-FA).
- POTO matching cost
	- Spatial priors helps (the center of prediction matched to GT cannot be outside of the GT box)
	- Balanced IoU and classification (by multiplication, better than summation)
- 3D Max Filtering (3DMF)
	- [CenterNet](centernet.md) uses 2D max filtering to replace NMS
	- Duplicate predictions majorly come from the nearby spatial regions of the most conf prediction, and comes from neighboring scales. As objects with sizes on the border of a stage may be automatically assigned to neighboring stage of the FPN. 
	- 3DMF is a module to perform 3D max pooling to provide sharper response. It is used as a differentiable post-processing step inside the network.
- Auxiliary loss to speed up convergence.

#### Technical details
- By using POTO and 3D MF, the scores of duplicate samples are significantly suppressed. 
- On CrowdHuman, the recall is even higher than the theoretical upper limit with GT (applying NMS on GT). 
- [MultiBox](https://arxiv.org/abs/1412.1441) is the first paper to propose bipartite matching between pred and GT, way earlier than [DETR](detr.md).

#### Notes
- [Review on Zhihu by 1st author](https://mp.weixin.qq.com/s/lfkcbfQrFOGUpkGB7Oh-FA)
	- About spatial prior in matching cost

	> 在α合理的情况下，空间先验不是必须的，但空间先验能够在匹配过程中帮助排除不好的区域，提升绝对性能；研究者在 COCO 实验中采用 center sampling radius=1.5，在 CrowdHuman 实验中采用 inside gt box. 理由很简单，CrowdHuman 的遮挡问题太严重，center 区域经常完全被遮挡。

