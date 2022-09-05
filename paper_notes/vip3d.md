# [ViP3D: End-to-end Visual Trajectory Prediction via 3D Agent Queries](https://arxiv.org/abs/2208.01582)

_September 2022_

tl;dr: Introduce MOT to [DETR3D](detr3d.md).

#### Overall impression
In a typical autonomous driving pipeline, perception and prediction modules are separated and they communicate via hand-picked features such as agent boxes and trajectories as interfaces. The **information loss** caps the performance upper bound of the prediction module. Also, **errors in perception module can propagate and accumulate**, and are almost impossible to correct in prediction. Last, the prediction is typically trained with perception GT, leaving a large gap between inference and training performance.

Previous efforts in joint perception and prediction include [FIERY](fiery.md) and [BEVerse](beverse.md), but ViP3D is the 1st paper to seriously borrows the SOTA approaches of prediction into the BEV perception task. ViP3D also explicitly models instance-wise agent detection, tracking and prediction in a fully differentiable fashion. [FIERY](fiery.md) and [BEVerse](beverse.md) both uses heatmap based method, and makes it inevitable to rely on heuristics and manual postprocessing, leaving these method not end-to-end differentiable.

The query-based methods seems to be the future of end to end 


#### Key ideas
- The queries are not instance level as in [DETR3D](detr3d.md) but rather track-level or agent-level, throughout multiple frames. 
- Evaluation metric of EPA (end-to-end prediction accuracy).

#### Technical details
- Trajectory prediction methods
	- Regression based method 
	- Goal based method (TNT)
	- Heatmap based method (DenseTNT, Home, Thomas)

#### Notes
- The missing ego pose
	- This paper was heavily inspired by [MOTR](motr.md), and extends [DETR3D](detr3d.md) into temporal domain.
	> The extension of [DETR3D](detr3d.md) to temporal domain is relatively straightforward, using the 3D reference point, transforming to the past timestamps using ego motion, and then project to the images from the past timestamps.

	- The valuable information of Ego pose was not introduced in this paper. In other words, the vehicles are tracked in the 3D ego coordinate, not the 3D world coordinate. ViP3D establishes a good baseline, but incorporating ego motion so everything is tracked in 3D world coordinate may be a direction for future improvement.
- The missing SOTA comparison. 
	- The paper still lacks comparison with other joint perception+prediction papers (at least with FIERY and BEVerse). 

