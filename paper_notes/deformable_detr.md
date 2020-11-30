# [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)

_October 2020_

tl;dr: Improved DETR that trains faster and performs better to small objects. 

#### Overall impression
Issues with [DETR](detr.md): long training epochs to converge and low performance at detecting small objects. DETR uses small-size feature maps to save computation, but hurt small objects. 

Deformable DETR first reduces computation by attending to only a small set of key sampling points around a reference. It then uses multi-scale deformable attention module to aggregate multi-scale features (without FPN) to help small object detection.

There are several papers on improving the training speed of [DETR](detr.md).

- [Deformable DETR](deformable_detr.md): sparse attention
- [TSP](tsp.md): sparse attention
- [Sparse RCNN](sparse_rcnn.md): sparse proposal and iterative refinement


#### Key ideas
- Efficient Attention
	- Pre-defined sparse attention patterns. 
	- Learn data-dependent sparse attention --> Deformable DETR belongs to this
	- Low rank property in self-attention
- Complexity of DETR
	- Encoder: self attention $O(H^2W^2C)$, quadratically with feature size.
	- Decoder: cross attention $O(HWC^2 + NHWC)$, linearly with feature size. Self-attention $O(2NC^2+N^2C)$

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

