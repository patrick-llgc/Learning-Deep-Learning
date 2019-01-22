# [Deep Reinforcement Learning for Vessel Centerline Tracing in Multi-modality 3D Volumes](https://link.springer.com/chapter/10.1007/978-3-030-00937-3_86)

_Jan 2019_

tl;dr: Use DRL for vessle centerline tracing

#### Overall impression
The paper proposes a new method for tracing using deep reinforcement learning (DRL). However the paper uses a heavily handcrafted reward function which may be simplified.

#### Background
- Old method for tracing blood vessel searches for a shortest shortest path with various handcrafted vessleness or medialness cost metrics. This may be suboptimal given anatomical anomaly or imaging artifact.  Alternatively supervised CNN segmentaion is time consuming as it searches for the entire 3D space.

#### Key ideas
- 3D tracking agent has 6 DOF in action space, {L, R, F, B, U, D}
- Distance from current point $p_t$ to a point on the centerline $g_d$

$$
D(p_t, G) = ||\lambda(p_t -g_{d+k}) + (1-\lambda) (g_{d+k+1} - g_{d+k-1})||
$$
- The second term in D is momentum enforcing he agent towards the direction of the curve. (Actually the frist term already has some momentum.)

- Reward is related to the D, where $l$ is the vessle radius empirically determined.

$$
r_t = D(p_t, G) - D(p_{t+1}, G), \text{ if } ||p_t - g_d|| <=l; \\
r_t = ||p_t - g_d|| - ||p_{t+1} - g_d||, \text{ otherwise.}
$$
- The paper uses standard DQN formulation, inlcuding periodic updates of  target network, exprience replay, \epsilon-greedy. The paper also added randomness of initialization by setting the intial point randomly along the center line, and randomly within a 10x10x10 voxel patch.
- In contrast enhanced CT, the DRL method without the momentum reward term performs the best. 

#### Technical details
- During tracing (inference), a momemtum on action value is used to curb oscillation in action, and a spline fit is used to fit the integer positions. 
- Input size $(2 \times 2 \times 2 \text{ mm}^3) \times (50 \times 50 \times 50)$.

#### Notes
- DRL should be applicable to rib tracing, and the contrast is closest to contrast enhanced. The reward function should be able to be simplified. Maybe start with a 2D toy exmaple first to expriement with the reward function.

