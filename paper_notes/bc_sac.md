# [BC-SAC: Imitation Is Not Enough: Robustifying Imitation with Reinforcement Learning for Challenging Driving Scenarios](https://arxiv.org/abs/2212.11419)

_June 2023_

tl;dr: IL + RL >> IL. IL lays the foundation and RL solidifies it.

#### Overall impression
The paper combines IL and RL to get the best of both worlds. 

**This paper optimizes the action, not trajectory.** Steer x acceleration = 31 x 7 = 217 actions.

Previous way to improve safety is to augment a learned planner with a non-learned fallback layer that guarantees safety.

SAC-BC still requires heuristically choosing the tradeoff between IL and RL objectives. 

#### Key ideas
- IL vs RL
	- IL is a simple and powerful way to leverage high quality human driving data, which can be collected at scale. It can generate human-like driving behavior. Yet IL alone fail to account for safety and reliability. 
	- RL methods trained in closed-loop, RL policies can establish **causal relationships** between observations, actions and outcomes. This generates policies that are aware of safety considerations that are only **implicit** in the human demos. RL improves safety and robustness, especially in rare and challenging scenarios in the absence of abundant data.
	- Relying on RL alone is problematic as it heavily depends on reward design, and this is still an open challenge. The results are typically unnatural. 
- IL + RL
	- IL increases realism and eases the reward design burden of RL.	- Combining IL and RL with simple reward functions, we can substantially improve the safety and reliability of policies learned over imitation alone without comprising on naturalistic human-like behavior. 
	- IL provides an abundant sources of learning signal without the need for reward design (sparse reward), and RL addresses the weakness of IL in rare and challenging scenarios when data is scarce. 
- RL uses a SAC baseline (soft actor-critic)
	- Reward: encode safety constraints. 
	- Collision: encourages the vehicle to keep a certain distance from nearby objects.
	- Off-road: encourages the vehicle to keep a negative distance to road edge.
	- Future directions may include trip-progress, traffic rule adherence, and passenger comfort. 
- Training: 
	- The performance of learning-based methods **strongly depends on the distribution of the training data**. This is a particularly important factor in settings with a long-tail distribution of safety-critical examples, such as autonomous driving.
	- training on more difficult examples results in better performance than using all the available data, both in aggregate and in challenging scenarios.
- Eval: slicing the dataset by difficulty.
	- The difficulty model is trained on 14k hours of expert trajectories in the same manner. It predicts whether a segment will result in a collision or near miss, labeled by humans. (see [IL Difficulty Model](https://arxiv.org/abs/2212.01375))
	- Then create top1, top10, top50 by selecting the most challenging clips.
	- Metrics: failure rates (no collision or offroad), route progress ratio.
- Closed-loop Simulation: non-reactive agents from log-sim of 10 seconds to mitigate pose divergence. 
- BC trained on top1 is significantly worse than that trained on top10 or All. That means BC relies on **large amount of data** to implicitly infer driving preference. Open loop BC tends to fall victim to distribution shifts when not provided with enough data. --> in this sense, all AI 1.0 such as object detection are BC.
- RL generates actions that deviates significantly from demo yielding unnatural and uncomfortable behavior. --> Cost optimization in traditional optimization/sampling based planning is also like this, and needs a gentle touch of IL to make it more human like.
- Progress-safety tradeoff

#### Technical details
- Trained on 100k miles data from Waymo's fleet. 6.4 Million clips for training and 10k for eval.
- ALVINN from 1988 was based on imitation learning.
- Open-loop IL suffers from **covariance shift** which can be addressed by closed-loop training.
- IL also lack explicit knowledge of what constitutes good driving, such as collision avoidance.
- Negative data is hard to collect to train an effective IL system at safety boundary. For drones maybe ("Learning to fly by crashing" from CMU), but not for cars. 
- Actions are not directly observable (*why not? cannot we just collect steer and acceleration during human driving?*), so actions are generated via optimization from matching the trajectory, using an vehicle dynamic model (kinematic bicycle model).
- MGAIL: closed loop training, and differentiability of simulator dynamics. 


#### Notes
- Questions and notes on how to improve/revise the current work
