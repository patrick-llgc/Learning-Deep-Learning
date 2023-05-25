# [ViP3D: End-to-end Visual Trajectory Prediction via 3D Agent Queries](https://arxiv.org/abs/2208.01582)

_September 2022_

tl;dr: Introduce MOT to [DETR3D](detr3d.md).

#### Overall impression
In a typical autonomous driving pipeline, perception and prediction modules are separated and they communicate via hand-picked features such as agent boxes and trajectories as interfaces. The **information loss** caps the performance upper bound of the prediction module. Also, **errors in perception module can propagate and accumulate**, and are almost impossible to correct in prediction. Last, the prediction is typically trained with perception GT, leaving a large gap between inference and training performance.

[ViP3D](vip3d.md) mainly addresses what information to pass between perception and prediction. Instead of explicitly hand-crafted features for each track, Vip3D uses information-rich track queries as the interface between perception and prediction, and reaches end to end differentiability.

Previous efforts in joint perception and prediction include [FIERY](fiery.md) and [BEVerse](beverse.md), but ViP3D is the 1st paper to seriously borrows the SOTA approaches of prediction into the BEV perception task. ViP3D also explicitly models instance-wise agent detection, tracking and prediction in a fully differentiable fashion. [FIERY](fiery.md) and [BEVerse](beverse.md) both uses heatmap based method, and makes it inevitable to rely on heuristics and manual postprocessing, leaving these method not end-to-end differentiable.

The query-based methods seems to be the future of end to end autonomous driving. It is borrowed in [UniAD](uniad.md). The query-centric information processing seems to be inspired by [Perceiver](perceiver.md), which was first proposed to break the scaling of computaiton with input size and save computational cost. 


#### Key ideas
- The queries are not instance level as in [DETR3D](detr3d.md) but rather track-level or agent-level, throughout multiple frames. 
- Evaluation metric of EPA (end-to-end prediction accuracy).
	- EPA is essentially a KPI to evaluate track to track matching.
- Part 1: Query-based tracker
	- The track queries are decoded to 3D reference points and projected to images to gather image features, then updated to predict queries of the next time stamp, decoded again to generate new 3D reference points. --> No explicit handling of ego pose here. This could be a further point of improvement. 
	- Query supervision: this is TALA (tracklet aware label assignment) in [MOTR](motr.md).
- Part 2: Trajectory Predictor
	- 3 components, agent encoding, map encoding and trajectory decoding. The agent encoding has been done by track queries, so only map encoding and trajectory decoding are left.
	- Map encoding with [VectorNet](vectornet.md). It interacts with track queries with cross attention Q = Attention(Q, M).
	- Trajectory decoder: 3 types are explored, in technical details below.


#### Technical details
- Trajectory prediction methods
	- Regression based method 
	- Goal based method (TNT)
	- Heatmap based method (DenseTNT, Home, Thomas)
- End to end learning of perception + prediction is better than baseline. The only difference between ViP3D and the baselines are the **interface between the tracker and the trajectory predictor**. 
	- Instead of passing implicit **track queries** to interact with Map feature M and then decoded as trajectories, the detached baseline passes **agent/track attributes and states**.
	- Image features + track features == Image features >> track features. Image features contains good enough information for trajectory predictor. 

#### Notes
- The missing ego pose
	- This paper was heavily inspired by [MOTR](motr.md), and extends [DETR3D](detr3d.md) into temporal domain.
	> The extension of [DETR3D](detr3d.md) to temporal domain is relatively straightforward, using the 3D reference point, transforming to the past timestamps using ego motion, and then project to the images from the past timestamps.

	- The valuable information of Ego pose was not introduced in this paper. In other words, the vehicles are tracked in the 3D ego coordinate, not the 3D world coordinate. ViP3D establishes a good baseline, but incorporating ego motion so everything is tracked in 3D world coordinate may be a direction for future improvement.
- The missing SOTA comparison. 
	- The paper still lacks comparison with other joint perception+prediction papers (at least with FIERY and BEVerse). 

