# [Jointly Learnable Behavior and Trajectory Planning for Self-Driving Vehicles](https://arxiv.org/abs/1910.04586)

_June 2024_

tl;dr: Learning BP and TP jointly for ML-based motion planner.

#### Overall impression
This paper describes a way to "learn" a mmotion planner. Yet in reality the learning happens only in the relative weights of the already hand-crafted cost functions.

The paper has a very nice introduction what is motion planning, and the two components behavior planning and trajectory planning. The paper gives a very good high level review of the motion planning problem without poluting the main text with arcane math formulae without too much information.

#### Key ideas
- Learn a cost function. This idea is not new, but rather from a paper "Maximum margin planning" (MMP) bridge the challenging leap from perception’s model to
costs for a planner. 
	- In practice, it is often done by hand-designed heuristics that are painstakingly validated by observing the resulting robot behavior. 
	- MMP proposes a novel method whereby we attempt to automate the mapping from perception features to costs. We do so by framing the problem as one of supervised learning to take advantage of examples given by an expert describing desired behavior.
- Loss formulation
	- max-margin loss: max-margin learning loss penalizes trajectories that have small cost and are different from the human driving trajectory
	- imitation loss: L2 loss to expert trajectory

#### Technical details
- Similarity to IRL (inverse reinforcement learning) in that they aim to infer the underlying cost or reward structure that an expert follows. 
	- In IRL, the goal is to find a reward function that explains the observed behavior
	- In max-margin loss learning, the objective is to ensure that the cost of the expert’s trajectory is lower than that of any other trajectory.

#### Notes
- Questions and notes on how to improve/revise the current work

