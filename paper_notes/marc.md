# [MARC: Multipolicy and Risk-aware Contingency Planning for Autonomous Driving](https://arxiv.org/abs/2308.12021)

_June 2024_

tl;dr: Generating safe and non-conservative behaviors in dense dynamic environment, by combining multipolicy decision making and contigency planning.

#### Overall impression
This is a continuation of work in [MPDM](mpdm.md) and [EUDM](eudm.md). It introduces dynamic branching based on scene-level divergence, and risk-aware contingency planning based on user-defined risk tolerance.

POMDP provides a theoretically sounds framework to handle dynamic interaction, but it suffers from curse of dimensionality and making it infeasible to solve in realtime. 

[MPDM](mpdm.md) and [EUDM](eudm.md) are mainly BP models, but [MARC](marc.md) combines BP and MP.

 belief trees heavily and decomposes POMDP into a limited number of closed-loop policy evaluations.

For the policy tree (or policy-conditioned scenario tree) building, we can see how the tree got built with more and more careful pruning process with improvements from different works. 

* [MPDM](mpdm.md) is the pioneering work prunes belief trees heavily and decomposes POMDP into a limited number of closed-loop policy evaluations. MPDM has only one ego policy over planning horizon (8s).
* [MPDM](mpdm.md) iterates over all ego policies, and uses the most likely one policy given road structure and pose of vehicle.
* [MPDM2](mpdm2.md) iterates over all ego policies, and iterate over (a set of) possible policies of other agents predicted by a motion prediction model.
* [EUDM](eudm.md) itrates all ego policies, and then iterate over all possible policies of other agents to identify **critical scenarios** (CFB, conditioned filtered branching). Guide branching in both action and intention space. [EPSILON](epsilon.md) used the same method.
* [MARC](marc.md) iterates all ego policies, iterates over a set of predicted policies of other agents, identifies **key agents** (and ignores other agents even in critical scenarios). 


All previous MPDM-like methods consider the optimal policy and single trajectory generation over all scenarios, resulting in lack of gurantee of policy consistency and loss of multimodality info.

#### Key ideas
- *Contigency planning* generates deterministic behavior for mulutiple future scenarios. In other words, it plans a short-term trajectory that ensures safety for all potential scenarios. --> This is very similar to the idea of *backup plan* in [EPSILON](epsilon.md).
- Scenario tree construction
	- generating policy-conditioned critical scenario sets via closed-loop forward simulation (similar to CFB in EUDM?).
	- building scenario tree with scene-level divergence assessment. Determine the latest timestamp at which the scenario diverge. Delaying branching time as much as possble.
		- State variables in trajectory optimization decreases
		- Smooth handling of different potential outcomes, more robust to disturbance (more mature driver-like).
- Trajectory tree generation with RCP
	- RCP (risk-aware contingency planning) considers tradeoff beween conservativeness and efficiency.
	- RCP generates trajectories that are optimal in multiple future scenarios under user-defined risk-averse levels. --> This can mimic human preference.
	- Risk tolerance levels of the users is controlled by a hyperpraram alpha.
- Evalution
	- Selection based on both policy tree and trajectory tree (new!), ensuring consistency of policies
- MARC are more robust under uncertain interactions and fewer unexpected policy switches
	- can handle cut-in with smoother decel, and can handle disturbance (prediciton noise, etc) 
	- with better effiency (avg speed) and riding comfort (max decel/acc).

#### Technical details
- Planning is hard from uncertainty and interaction (inherently multimodal intentions). 
	- For interactive decision making, MDP or POMDP are mathematically rigorous formulations for decision processes in stochastic environments. 
	- For static (non-interactive) decision making, the normal trioka of planninig (sampling, searching, optimization) would suffice.

#### Notes
- Questions and notes on how to improve/revise the current work

