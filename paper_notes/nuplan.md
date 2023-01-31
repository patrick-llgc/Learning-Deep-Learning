# [nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles](https://arxiv.org/abs/2106.11810)

_January 2023_

tl;dr: closed-loop benchmark for ML-based motion planners.

#### Overall impression
This paper is one pioneering paper/tech report on **ML-based motion planner (neural planner)**. The report points out that in order to get a fair benchmark, closed loop simulation has to be used.

> Background: AV perception has witnessed impressive progress. In contrast, existing solutions for AV planning are primarily based on carefully engineered expert systems. They require significant amount of work to adapt to new geographies and does not scale with training data. Having ML-based planning will pave the way to a full Software 2.0 stack.

> Prediction focuses on the behavior of other agents, while planning relates to the ego vehicle behavior. Prediction is multimodal, and for each agent we predict the N most likely trajectories. Planning is unimodal.

#### Key ideas
- Planning metrics
	- Traffic rule violation
	- Human driving similarity
	- Vehicle dynamics
	- Goal achievement
	- Scenario based
- Open loop vs closed loop

#### Technical details
- Prediction datasets
	- Argoverse: 320 h
	- NuPredict: 5 h
	- Lyft: 1118 h
	- Waymo: 570 h
- Simulation
	- Carla
	- AirSim
- Problems with prediction datasets --> Not super strong arguments, but even if these are not drawbacks for prediction datasets, it does not invalidate the need for a planning dataset.
	- lack of high level goals --> This is not observable for other agents
	- choice of metrics: displacement metrics does not take into account multimodal nature --> actually winner-takes-all does this.
	- open loop --> 3-8 sec may be good enough
- Open loop evaluation with L2-based metrics are not suitable for fairly evaluating long term (beyond 3-8 seconds) planners. Lack of closed-loop evaluation leads to systematic drift. --> Is 3-8 seconds good enough though?

#### Notes
- Questions and notes on how to improve/revise the current work
