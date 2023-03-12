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
- Scenarios in the planning metrics
	- merges, lane changes
	- protected or unprotected left or right turns
	- interaction with cyclists, interaction with pedestrians at crosswalks or elsewhere, interactions with close proximity or high acceleration
	- double parked vehicles
	- stop controlled intersections
	- driving in construction zones.
- Open loop vs closed loop
	- open loop: no interactions are considered, it is NOT used to control the vehicle position.
	- closed loop: proposed trajectory is used as a reference for a controller, and the planning system is gradually corrected at each timestep with the new state of the vehicle. 
- Closed-loop can be non-reactive or reactive. In reactive closed-loop, all other agents will have a planning model and reacts to ego.
- In closed loop environment, *sensor data warping* or *novel view synthesis* CAN be done to better reflect what the perception system takes as input. To better focus on planning, nuPlan assumes a perfect (or uniformly performing) perception system and does not do sensor data warping.

![Open Loop](https://woven-planet.github.io/l5kit/_images/open-loop.svg)
![Closed Loop](https://woven-planet.github.io/l5kit/_images/closed-loop.svg)

#### Technical details
- Prediction datasets
	- Argoverse: 320 h
	- NuPredict: 5 h
	- Lyft: 1118 h
	- Waymo: 570 h
- Code is containerized for portability in order to enable closed-loop evaluation on a secret test set. 
- Simulation (Carla, AirSim) enabled breakthroughs in planning and RL.
- Problems with prediction (motion forecasting) datasets
	- lack of high level goals --> This is not observable for other agents
	- choice of metrics: displacement metrics does not take into account multimodal nature --> actually winner-takes-all handles this to a certain extent, but not entirely.
	- open loop --> 3-8 sec may be good enough. 
- Open loop evaluation with L2-based metrics are not suitable for fairly evaluating long term (beyond 3-8 seconds) planners. Lack of closed-loop evaluation leads to systematic drift. --> Is 3-8 seconds good enough for planning though?
- ML-based planning
	- This field has yet to converge

#### Notes
- The three issues with existing prediction dataset are not super strong arguments, but even if these are not drawbacks for prediction datasets, it does not invalidate the need for a planning dataset.
- [Lyft1001 dataset (CoRL2020)](https://arxiv.org/abs/2006.14480) has a good explanation of open loop vs closed loop [in this page](https://woven-planet.github.io/l5kit/planning_open_loop.html) with a [Youtube Video](https://www.youtube.com/watch?v=Jygsh17QbxY&t=689s)

