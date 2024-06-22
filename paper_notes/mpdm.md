# [MPDM: Multipolicy decision-making in dynamic, uncertain environments for autonomous driving](https://ieeexplore.ieee.org/document/7139412)

_June 2024_

tl;dr: A principled decision making framework to account for interaction with other agents. 

#### Overall impression
MPDM starts with a regorous formulation and makes assumptions with domain knowledge in autonomous driving to simplify the problem to be tractable for online deployment. 

MPDM forward simulates multiple scenarios initiated by different policy (high level intention or behavior pattern) of ego and how other vehicles would react. This brings two advantages of MPDM.

* MPDM can enable **personalized driving experience** we can evaluate multipe outcome using user defined cost function to accomodate different driving preferences. --> This is extended to include risk tolerance in [MARC](marc.md).

* MPDM can enable intelligent and human-like behavior of **active cut-in** into dense traffic flow even when there is not a sufficient gap present(华山挤出一条路). This is NOT possible with a predict-then-plan schema without considering the interaction explicitly.

Despite simple design, MPDM is a pioneering work in decision making, and improved by subsequent works. MPDM has the assumption that the ego intention does not change within the planning horizon (10s, at 0.25s). This is improved by [EUDM](eudm.md) which allows change of ego policy within planning horizon once, and [MARC](marc.md) which introduces risk aware contigency planning.

#### Key ideas
- Assumptions
	- Much of decision making made by human drivers is over discrete action. --> This is largely true, but the discreteness may get blurry when in dense urban areas.
	- Other vehicles will make reasonable safe decisions. 
- MPDM models vehicle behavior as closed-loop policy for ego AND nearby vehicles.
- Approximation
	- Ideally we want to sample high likelihood senarios on which to make decisions, and to focus sampling on more likely outcomes
	- Choose policies from a finite fixed set for ego and other agents.
	- Approximate interaction with deterministic closed-loop simulation. Given a sampled policy and the driver model, the behaviors of other agents are deterministic.
	- The decoupling of vehicle behavior as the instantaneous behavior is independent of each other.
	- The formulation is highly inspiring and is the foundation of [EPSILON](epsilon.md) and all follow-up works.
	- The horizon is 10s with 0.25s timesteps, so a 40-layer deep tree. 

#### Technical details
- MPDM How important is the closed-loop realism? The paper seems to argue that the inaccuracy in closed-loop simulation does not affect final algorithm performance that much. Close-loop or not seems to be the key.


#### Notes
- The white paper from [May Mobility](https://maymobility.com/resources/autonomy-at-scale-white-paper/) explains the idea with more plain language and examples. 