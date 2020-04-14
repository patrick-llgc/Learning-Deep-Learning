# [NMP: End-to-end Interpretable Neural Motion Planner](http://www.cs.toronto.edu/~wenjie/papers/cvpr19/nmp.pdf)

_April 2020_

tl;dr: Learn a cost volume for planning and use diverse sampling to find the best trajectory.

#### Overall impression
**Prediction (motion forecasting)** tackles the problem of estimating the future positions of all actors as well as their intentions (changing lanes, parking). Finally, **motion planning** takes the output from previous stacks and generates a safe trajectory fort he SDV to execute via a control system.

**Ego-Motion forecasting** provides strong cue on how the SDV will move in future. However it may not use the information of the dynamic environment, i.e., how other cars will move. 

The author argues that traditional metric (such as mAP) for perception may not be optimal. Such metrics weigh all actors uniformly, whereas nearby actors impact downstream modules more. However, large companies in industry still favors decoupled stack where large engineering teams work in parallel with specific task-specific objectives in mind. Advances in upstream stack may not necessarily translates to overall system improvement. (I guess that is part of the responsibility of engineering management, to identify bottlenecks in the entire engineering stack.) Therefore the authors (Uber ATG) advocates for end-to-end systems.

The authors still uses a perception loss, but this loss is only used to guide the system and provide interpretability to the end-to-end stack.

This work is based on [Fast and Furious](faf.md) and [IntentNet](intentnet.md). It uses similar input of lidar data and semantic maps. 


#### Key ideas
- Why cost-volume?
	- The ultimate output of the neural network is a cost volume of the size HxWxT. This cost volume represents the "goodness" of each possible location that the SDV can take within the planning horizon. The non-parametric cost volume **allows uncertainty and multimodality**.
	- Sampling: space sampling based on clothoids], and then time sampling to determine the space-time profile. A set of trajectories is generated and evaluated against a predefined cost. This is a simple look-up operation and can be done **efficiently**.
	$$s^* = \arg \min_s \sum_t c^t (s^t) $$
	- Imitation learning approaches has to design multi-modality anchors specifically. (cf [Rules of the road](ror.md), [MultiPath](multipath.md), [MultiPath Uber](multipath_uber.md)).
- Loss for training motion planner:
	- Max-margin loss. Encourage the GT to have the lowest cost. Sample negative path, and the loss is c(gt) - c(pred). 
	- Distance loss between gt and pred
	- Traffic violation loss is a constant if negative violates traffic rules. 

#### Technical details
- 10 previous frames (1 second history) concatenated along with z dimension. (ZT)xWxH.


#### Notes
- Optimization based planners: takes in perception and prediction results and formulate an optimization problem based on manually engineered cost function.
- A [clothoid](https://en.wikipedia.org/wiki/Euler_spiral)'s curvature begins with zero at the straight section (the tangent) and increases linearly with its curve length. See [this video](https://www.youtube.com/watch?v=rvKVFyR7XVc) to get a flavor on how it is used in path planning.
- [Basic facts about Clothoids](https://mse.redwoods.edu/darnold/math50c/CalcProj/Fall07/MollyRyan/Paper.pdf)