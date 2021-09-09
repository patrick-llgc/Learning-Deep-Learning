# [MP3: A Unified Model to Map, Perceive, Predict and Plan](https://arxiv.org/abs/2101.06806)

_September 2021_

tl;dr: Mapless driving from lidar.

#### Overall impression
This paper continues the line of work of end-to-end self-driving by Raquel's team. 

Mapless driving can 1) serve as the fail-safe in the case of localization failures or outdated maps, and 2) potentially unlock self-driving at scale at a much lower cost.

Without map, the search space to plan a safe maneuver from the SDV goes from narrow envelopes around the lane center lines to the full set of dynamically feasible trajectories.

#### Key ideas
- Online map
	- drivable area
	- reachable lanes (unsigned distance to the closest reachable lane centerline)
	- intersections (traffic is controlled via traffic lights or traffic signs)
- dynamic occupancy field = initial occupancy + temporal motion field (11 x 0.5 = 5 s)
	- which space is occupied by dynamic objects and how do they move over time.
	- computation is agnostic to the number of objects.
	- usually motion forecasting involve unsafe discrete decisions such as confidence thresholding and NMS.
	- P3 proposed the idea of dynamic occupancy field
- Motion planning has 3 goals: safe, comfortable and progressing toward the goal
- Trajectory sampling
	- sampled from large scale dataset of real trajectories
	- To generate continuous velocity and steering, instead of directly using the trajectories, use the acc and steering rate profiles to rollout a bicycle model from the initial SDV state.
- Route prediction
	- The network takes in a driving command and output a dense probability map R in BEV.
	- A driving command is characterized by a tuple (a, d) where a = {keep, left, right} and d is longitudinal distance to action. This is similar to command given by the GPS
- Trajectory scoring
	- The trajectory needs to overlap high with route map R.
	- The trajectory needs to stay on the road and avoid encroaching onto the sidewalks or curbs.
	- The trajectory needs to be safe, and gets penalized for overlapping with dynamic grids.
	- The trajectory needs to be comfortable and gets penalized for jerk, lateral accel, curvature and curvature change rates.

#### Technical details
- Lidar input formulation follows that of [FaF](faf.md), [intentNet](intentnet.md) and [Pixor](pixor.md), with 0.2 m/voxel, with motion compensation.
- Two-stage training: train online mapping, dynamic occupancy field and routing.

#### Notes
- [5 min video on youtube](https://www.youtube.com/watch?v=8LPrYcWZaRc), [Best paper candidates](http://cvpr2021.thecvf.com/node/290)