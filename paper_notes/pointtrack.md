# [PointTrack: Segment as Points for Efficient Online Multi-Object Tracking and Segmentation](https://arxiv.org/abs/2007.01550)

_July 2020_

tl;dr: Use [PointNet](pointnet.md) to extract embedding vectors from instance segmentation masks for tracking.

#### Overall impression
This work tackles the newly created track of [MOTS (multiple object tracking and segmentation)](https://arxiv.org/abs/1902.03604) <kbd>CVPR 2019</kbd>. It uses tracking-by-segmentation. It uses existing work of instance segmentation, and the main contribution lies in the association/tracking part. 

[PointTrack](pointtrack.md) uses a single stage instance segmentation method with a seeding location. This makes it compatible with many instance segmentation method, such as [CenterMask](centermask.md) or [BlendMask](blendmask.md).

[deepSORT](deep_sort.md) extracts features from image patches, while [PointTrack](pointtrack.md) extracts features from 2D point cloud. 

#### Key ideas
- Instance segmentation with [SpatialEmbedding](spatial_embedding.md)
- Tracking Architecture
	- Regard 2D image pixels as un-ordered 2D point clouds and learn instance embeddings.
	- Get context-aware patch: Dilate bbox by 20% to include more environment
	- Uniformly sample 1000 points on the object an 500 points on the background
	- Use color, position offset from center, and class category to encode each 2D cloud point.
		- Position is lifted from 4-dim to 64 dimension following [Transformer](transformer.md) to make it easier to learn.
- Association of instances across frames is done with embedding distance and segmentation mask IoU. 
- Seed consistency:
	- Using Optical flow and last seed to encourage consistent seed. --> This additional optical flow map will definitely help [CenterTrack](centertrack.md).
	- Penalize difference between the warped seed from last frame with optical flow and the seed predicted from current frame. 
- Points with highest (top 10%) importance can be visualized by their weights, a natural feature from [PointNet](pointnet.md) embedding.
- Visualization of instance embedding with T-SNE is also quite interesting.

#### Technical details
- Ablation study showed that the removal of color in the input leads to the biggest drop.

#### Notes
- Questions and notes on how to improve/revise the current work  

