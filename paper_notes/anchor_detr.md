# [Anchor DETR: Query Design for Transformer-Based Object Detection](https://arxiv.org/abs/2109.07107)

_September 2022_

tl;dr: Use encoded anchor to explicitly guide the detection queries to enable more focused attention and detection.

#### Overall impression
The paper is one of the series of papers to improve the convergence rate of [DETR](detr.md).

- [Deformable DETR](deformable_detr.md)
- [Anchor DETR](anchor_detr.md)
- [Conditional DETR](conditional_detr.md)

The object queries in the original [DETR](detr.md) is a set of learned embedding, and does not have physical meaning. The object queries do not focus on a specific region, making it hard to optimize. In DETR, object queries from different positions **predict objects collaboratively**.

[Anchor DETR](anchor_detr.md) is concurrent with [Conditional DETR](conditional_detr.md), and the ideas are roughly the same. Anchor DETR uses encoded anchors as the queries, and conditional DETR uses encoded anchors as positional embeddings and the object queries are still learned. 


#### Key ideas
- Encode the anchor points as object queries. A small MLP with two linear layers are used to project anchor points to object queries. In this way, the prediction slots (collection of allocated GT to a particular query) are more related to a specific position than DETR.
- Each anchor point can predict multiple object. --> This idea seems to stem from [CrowdDet](crowd_det.md) which can perform multiple detections per anchor for crowd detection.
	- Pattern embedding to differentiate objects with diff patterns (aspect ratio, size) at each anchor position. (N_p is very small, ~3). The patterns are shared. Pattern embedding is tiled for all positions to keep tranlational invariance and are added to object queries.
	- The pattern embedding does not help with original DETR without explicit anchor encoding.
- RCDA (row column decouple attention)
	- Saves memory while achieving similar performance with vanilla attention. Other types of efficient transformer seems to degrade DETR's performance.
- Anchor types: learned and uniform grid. Both perform similarly, but the learned points are more data driven. The learned anchor points roughly distributes uniformly, most likely due to the uniform distribution of objects in COCO dataset. --> Note that the **learned anchor points are different from learned queries**, where the learned anchors still perform as strong guide for network to focus.

#### Technical details
- [Deformable DETR](deformable_detr.md) is similar to two-stage detector as the position of sample point is stochastic to the hHW so it is not RAM (random access of memory) free.

#### Notes
- Questions and notes on how to improve/revise the current work
