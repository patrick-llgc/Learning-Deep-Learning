# [EUDM: Efficient Uncertainty-aware Decision-making for Automated Driving Using Guided Branching](https://arxiv.org/abs/2003.02746)

_June 2024_

tl;dr: A better MPDM with guided branching in both action and intention space.

#### Overall impression
In order to make POMDP more tractable it is essential to incorporate domain knowledge to efficiently make robust decisions (accelerate the problem-solving).

[MPDM](mpdm.md) reduces POMDP to closed-loop evaluation (forward simulation) of a finite discrete set of semantic level policies, rather than performing evaluaton for every possible control input for every vehicle (curse of dimensionality).

In EUDM, ego behavior is allowed to change, allowing more flexible decision making than MPDM. This allows EUDM can make a lane-change decision even before passing the blocking vehicle (accelerate, then lane change).

![](https://pic3.zhimg.com/80/v2-a7778368cbf39f083ef5ad5a2f931a4e_1440w.webp)


EUDM does guided branching in both action (of ego) and intention (of others).

EUDM couples prediction and planning module. 

It is further improved by [MARC](marc.md) where it considers risk-aware contingency planning.

#### Key ideas
- DCP-Tree (domain specific closed-loop policy tree), ego-centric
	- Guided branching in action space
	- Each trace only contains ONE change of action (more flexible than MPDM but still manageable). This is a tree with pruning mechanism built-in. [MCDM](mcdm.md) essentially has a much more aggressive pruning as only one type of action is allowed (KKK, RRR, LLL, etc)
	- Each semantic action is 2s, 4 levels deep, so planning horizon of 8s.
- CFB (conditional focused branching), for other agents
	- conditioned on ego intention
	- Pick out the potentially risky scenarios using **open loop** safety assement. (Open loop ignores interaction among agents, and allows checking of how serious the situation wil be if surrounding agents are completely uncoorpoerates and does not react to other agents.)
	- select key vehicles first, only a subset of all vehicles. --> Like Tesla's AI day 2022.
- Forward simulation
	- IDM for longitudinal simulation
	- PP (Pure pursuit) for lateral simulation
- EUDM output the best policy represented by ego waypoints (0.4s apart). Then it is sent to motion planner (such as [SCC](scc.md)) for trajectory generation.

#### Technical details
- Summary of technical details, such as important training details, or bugs of previous benchmarks.

#### Notes
- What are the predictions are fed into MP alongside the BP results from EUDM?

