# [MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction](https://arxiv.org/abs/1910.05449)

_April 2020_

tl;dr: Multi-modal behavior prediction via anchor trajectory with 6-second horizon. 

#### Overall impression
Behavior prediction is **inherently stochastic** as it is impossible to know what the agent may do next. Most previous method, including [Fast and furious](faf.md), [IntentNet](intentnet.md) and [ChauffeurNet](chauffeurnet.md) only predict MAP trajectory. [Rules of the Road](ror.md) predicts multiple future trajectories but through a set of unweighted samples. Sample-based generative methods have drawbacks: non-deterministic, hard to estimate errors, no way to perform probabilistic inference (e.g. to know the probability of collision in a space-time region). Also sample based approaches requires repeated inference to obtain multi-modal prediction.

Anchor trajectories are obtained by grouping logged trajectories (modes) in collected data, and provide templates for coarse granularity features for an agent. This idea brilliantly solved the exchangeability issue in multiple future prediction, as detailed in [Rules of the Road](ror.md).

MultiPath also used the semantic map representation used in previous methods such as [IntentNet](intentnet.md) and [ChauffeurNet](chauffeurnet.md) and [Rules of the Road](ror.md). 

[IntentNet](intentnet.md) also predicts intention. But they mainly focus on an MAP trajectory. [IntentNet](intentnet.md) only predict one set of trajectories and make it unsuitable for multiple future path prediction. This can be changed to predict multiple path, each per intent, and then during inference we can sample K most likely trajectory each associated with the top intent. The discrete intent prediction roughly corresponds to the discrete anchors in MultiPath, but anchor design is more data driven and flexibile. 


#### Key ideas
- The overall architecture is faster RCNN like
	- Agent centric network that can be applied to each agent uniformly. The architecture is Faster-RCNN like, with RoIAlign layer to extract agent specific features. 
	- Anchor classification (intent prediction) and waypoint offset prediction
	- Loss only cares about waypoints. Not too much about size and heading of the agent. 
- **Mixture Density Networks (MDN)** with log likelihood loss, similar to [Gaussian YOLOv3](gaussian_yolov3.md). This formulation helps with multiple future prediction. 
- K x T x 5 predictions. K anchors, T waypoints per trajectory (time steps) and 5 predictions per waypoint (x, y, std x, std y, p). --> Maybe K x (T x 5 + 1) as only one softmax logits per anchor?

#### Technical details
- Evaluation:
	- Log likelihood (LL) of the path given the image. Product of likelihood of all waypoints. 
	- Distance based metric (most of them are in L2 norm)
		- ADE: average displacement error
		- FDE: final displacement error
		- minADE: ADE of the closest trajectory to GT out of M trajectories, so that reasonable predictions that do not happen to be the GT do not get penalized
		- minSDE: squared version of minADE
- nat: or nit, is a bit equivalent with e as base. It is the unit for entropy.
- Stratified sampling to tackle imbalanced data	

#### Notes
- This paper proves again the brilliant idea of anchor. This is a tractable way to make an unknown number of predictions, which alleviates the exchangeability issue. Essentially we have to sort the GT in a structured way for prediction.
- [Multi-Modal Trajectory Prediction of Surrounding Vehicles with Maneuver based LSTMs](https://arxiv.org/abs/1805.05499) <kbd>IV 2018</kbd> uses 6 maneuver classes, and encode the ego car and neighboring car status as input, without map info. The loss is very similar to that of [multipath](multipath.md)
- The NLL loss is the same as used here [Uncertainty-aware Short-term Motion Prediction of Traffic Actors for Autonomous Driving](https://arxiv.org/pdf/1808.05819.pdf) <kbd>WACV 2020</kbd>