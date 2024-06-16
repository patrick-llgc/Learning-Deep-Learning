# [Baidu Apollo EM Motion Planner](https://arxiv.org/abs/1807.08048)

_June 2024_

tl;dr: An optimization-based motion planner with path-speed decoupled method.

#### Overall impression
Apollo EM motion planner is a scalable and easy-to-tune framework to handle traffic rules, obstacle decision and smoothness.

EM planner significantly reduces computational complexity by transforming a three-dimensional station-lateral-speed problem into two two- dimensional station-lateral/station-speed problems.

The paper does have some drawbacks, such as limited discussion on handling highly dynamic environments with unpredictable behaviors.

#### Key ideas
- Motion planned path should be safe and smooth.
- Frenet frames with time (SLT) to redue planning dimension with the help of a reference line. This is mainly true for high-speed driving scenario where lat and long are very different.
- Optimal trajectory in Frenet frame is a 3D constraint optimization problem
	- Direct 3d optimization
	- Decouple path and speed and break 3d optimization into two 2d optimization tasks.
- Path-speed approch can be suboptimal with appearance of dynamic objects. --> We need joint spatiotemporal optimization.
- Decision (behavior planning) is described by a rough and feasible trajectory. Then a convex space is generated based on the rough trajectory for further optimization.
- Architecture
	- A RL is generated for each lane
	- A frenet frame is genearted for each lane
	- Lane-level optimizer of trajectory in each frenet frame
	- Trajectory decider to select best trajectory
- Lane level Optimization of trajectory
	- E: SL projection
		- Static and low speed and oncoming traffic. The appearance of dynamic obstacles during path optimization will eventually lead to nudging, thus  high speed objects are NOT projected into SL graph.
		- Interaction is defined as ego and other obstacles bbox overlapping.
	- M: Path planning
		- DP: spline sampling
		- QP: convex optimization
	- E: ST projection
		- Static obstacles, low-speed, high-speed and oncoming vehicles are all considered.
		- Interaction defined as bbox overlapping. Only "relevant" obstacles are projected into ST graph. Relevant means it is "close enough". In other words, ST graph is the projection of a thin slice of SLT, but the thickness along L-dim is relatively thin.
		- For example, in Fig5, if a car starts cutting in at 2s, then the obstacle only appears at 2s. 
	- M: speed planning
		- DP: searching
		- QP: convex optimizaiton 
- The two M (optimization) stage in SL and ST shares many similarities and can be discussed under the same framework. 
- DP (dynamic programming) and QP (quaratic programming)
	- DP is decision, convert (prune) non-convex space into convex space.
	- QP is optimization, solve in the above convex space.
	- The interface between DP and QP is a DP path and a feasible tunnel (convex space). QP takes in the two and optimize it with vehicle dynamics and other constraints. 
- The solution space in SLT is highly non-convex.
	- in space, whether to nudge from left or right
	- in time, whether to pass or yield
- Cost design
	- DP cost
		- road (centerline guidance in SL, or ref speed guidance in ST)
		- obstacle
		- smoothness (heading, curvature, curvature derivative for SL, acceleration and jerk for ST)
	- QP cost
		- DP-solution guidance
		- smoothness (same as DP)

#### Technical details
- Range coverage of trajectory should be 8 sec or 200 meters. 
- Reaction time within 100 ms, compared to 200-300 ms reaction time for a normal human driver. 
- Sampling is favored to a search algorithm
	- computational resources: search space will be very large expanding multiple lanes
	- complexity in applying traffic regulations: traffic regulations will be per lane
	- maintaining stable and consistent trajectories.
- Spline: piece wise 5th order polynomial (quintic)

#### Notes
- The paper did not mention how reference lines are generated. Most likely hybrid A-star?
- How is multimodal prediciton handled in ST-graph?
- [Explanation by first author 樊昊阳](https://zhuanlan.zhihu.com/p/199719517)