# [SpatialEmbedding: Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth](https://arxiv.org/abs/1906.11109)

_July 2020_

tl;dr: Single stage instance segmentation with bottom-up approach.

#### Overall impression
Overall performance is not that great compared to other approaches. This forms the foundation of [PointTrack](pointtrack.md).

[PointTrack](pointtrack.md) uses a single stage instance segmentation method with a seeding location. This makes it compatible with many instance segmentation method, such as [CenterMask](centermask.md) or [BlendMask](blendmask.md).

The visualization of instance distance map looks great. 
- ![](https://raw.githubusercontent.com/davyneven/SpatialEmbeddings/master/static/teaser.jpg)

#### Key ideas
- SpatialEmbedding predicts 
	- a seed map (similar to the heatmap in [CenterNet](centernet.md) or [FCOS](fcos.md)
	- a sigma map to predict clustering bandwith (learned, largely proportional to bbox size)
	- offset map for each pixel pointing to the instance center

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

