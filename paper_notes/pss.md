# [PSS: Object Detection Made Simpler by Eliminating Heuristic NMS](https://arxiv.org/abs/2101.11782) 

_January 2021_

tl;dr: Add positive sample selector (PSS) branch to FCOS to achieve end-to-end detection.

#### Overall impression
[PSS](pss.md) keeps [FCOS](fcos.md)'s structure and training scheme as much as possible, and introduced one positive sample selector branch.

[PSS](pss.md) uses a simple binary classification head to enable selection of one positive sample for each instance, while [DeFCN](defcn.md) uses 3D max filtering to aggregate multi-scale features to suppress duplicated detection.

#### Key ideas
- PSS branch: one additional head attached to regression head.
- PSS loss. Cross entropy loss between GT and pred of $\hat{P}= \sigma(\text{cls}) \sigma(\text{centerness}) \sigma(\text{pss})$.
	- Still train FOCS with one-to-many label assignment. This helps convergence as [DeFCN](defcn.md) still uses it as auxiliary loss.
- Stop gradient operation to reconcile the conflict between PSS classification loss and the original FCOS loss.
	- This is in theory equivalent to training the original FCOS until convergence and freezing FCOS, and then training the PSS head only until convergence (thus PSS is a learnable NMS). In practice, two-step training leads to slightly worse performance.

#### Technical details
- In [FCOS](fcos.md), classication head and regression head are sibling heads. This is different from [CenterNet](centernet.md)
- Attaching the PSS head to classification head leads to worse performance. This is similar to that centerness head needs to be attached to regression head too, as observed in [FCOS](fcos.md).

#### Notes
- [Review by 1st author on Zhihu知乎](https://zhuanlan.zhihu.com/p/347515623)
- [Code on github](https://github.com/txdet/FCOSPss)

