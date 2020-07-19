# [PointTrack++ for Effective Online Multi-Object Tracking and Segmentation](https://arxiv.org/abs/2007.01549)

_July 2020_

tl;dr: Follow-up work of [PointTrack](pointtrack.md) for MOTS.

#### Overall impression
Three main contributions:


#### Key ideas
- Semantic segmentation map as seed map in [PointTrack](pointtrack.md) and [SpatialEmbedding](spatial_embedding.md).
- Copy and paste data augmentation for crowded scenes. Need segmentation mask.
- Training instance embedding:
	- [PointTrack](pointtrack.md) consists of D track ids, each with three crops with equal temporal space. It does not use 3 consecutive frames to increase the intra-track-id discrepancy. The space S is randomly chosen between 1 and 10.
	- [PointTrack++](pointtrack++.md) finds that for environment embedding, making S>2 does not converge, but for foreground 2D point cloud a large S (~12) helps to achieve a higher performance. Thus the embeddings are trained separately. Then the individual MLP weights are fixed, and a new MLP is trained to aggregate these info together. 

#### Technical details
- Image is upsampled to twice the original size for better performance.

#### Notes
- Questions and notes on how to improve/revise the current work  

