# [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/pdf/1706.02413.pdf)

_Feb 2019_

tl;dr: Use multistage PointNet to enable hierarchical learning.

#### Overall impression
The paper extends the work of pointnet from multi-stage/hierarchical viewpoint. It is impressive to see the authors uses multi-scale technique to address the loss of robustness in vanilla PointNet++.

#### Key ideas
- The original PointNet does not capture local features as each point is processed and aggregated into global features directly. This is like using a fully connected layer. This paper introduced hierarchical learning of features, resembling the learning process of CNNs.

#### Technical details
- Summary of technical details

#### Notes
- Point cloud can be complete or partial scan (from only one view point, generally the case for autonomous driving).
- The work is reviewed by the author Charles Qi in [this video](https://www.youtube.com/watch?v=Ew24Rac8eYE).