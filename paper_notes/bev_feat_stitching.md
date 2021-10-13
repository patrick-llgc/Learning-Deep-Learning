# [BEV-feat-stitching: Understanding Bird's-Eye View Semantic HD-Maps Using an Onboard Monocular Camera](https://arxiv.org/abs/2012.03040)

_January 2021_

tl;dr: predict BEV semantic maps from a single monocular video.

#### Overall impression
Previous SOTA [PyrOccNet](pyroccnet.md) and [Lift splat shoot](lift_splat_shoot.md) studies how to combine synchronized images from multiple cameras into a coherent 360 deg BEV map. [BEV-feat-stitching](bev_feat_stitching.md) try to stitch monocular video into a coherent BEV map. This process also requires knowledge of the camera pose sequence. 

The mapping of the intermediate feature map resembles that of [feature-metric mono depth](feature_metric.md) and feature-metric distance in [3DSSD](3dssd.md).

To be honest the results do not look as clean as [PyrOccNet](pyroccnet.md). Future work may be to combine these two trends, from both [BEV-feat-stitching](bev_feat_stitching.md) and [PyrOccNet](pyroccnet.md).

This paper has a follow-up work [STSU](stsu.md) for structured BEV perception.

#### Key ideas
- Takes in mono video as input
- BEV temporal aggregation module
	- Project the features to BEV space
	- BEV aggregation (BEV feature stitching) with camera pose.
		- Aggregation is done in a unified BEV grid (extended BEV)
- Intermediate feature supervision in camera space with reprojected BEV GT
	- Single frame object supervision 
	- Multiple frames static class

#### Technical details
- 200x200 pixels, 0.25 m/pixel, 50m x 50m
- The addition of dynamic classes helps with the static classes.

#### Notes
- The evaluation is still in mIoU, treating the problem as a semantic segmentation issue. However we perhaps should introduce the idea of instance segmentation for better prediction and planning.
- Stitching may have some noise with extrinsics and pose estimation and deep learning helps smooths this out.
