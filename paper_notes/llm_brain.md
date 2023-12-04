# [LLM-Brain: LLM as A Robotic Brain: Unifying Egocentric Memory and Control](https://arxiv.org/abs/2304.09349)

_December 2023_

tl;dr: A multi-LLM pipeline for active exploration of new environments.

#### Overall impression
LLM-brain is a multi-LLM pipline that acts as a robotic brain to unify ecocentric memory and control. The multiple agents communicate by natural language, provideing excellenet explanability.

Active exploration: extensively explore the unknown enviromenmt within a limited number of actions.

Reinforcement learning have dominated the field of embodied AI for many years until LLM in the field of Internet AI emerges. 

The results of LLM-brain is not that impressive, but it provides a reasonable starting framework for active exploration.

#### Key ideas
- Three agents
	- Eye: VQA model (BLIP-2), takes in images and questions. 
	- Nerve: ask questions to Eye, and provide summary to Brain
		- Why do we need nerve? We need to summarize the action and environment. "I move forward, and I see a sofa". This is inspired by [ChatCaptioner](chat_captioner.md).
	- Brain: takes in high level human instructions, memorize previous action and (potentially partial) environment observations, and infers next action based on its world knowledge.

#### Technical details
- Embodied AI tasks
	- Embodied QA (VQA)
	- Active exploration: "explore the house as completely as possible", evaluted by the explored areas vs total area.
	- LM-nav: vision-language navigation
- [PointNav task](https://aihabitat.org/challenge/2021/)
- Dataset: matterport3D.
- On the ask of active exploration, LLM-brain provides a slightly better performance than the random baseline.

#### Notes
- Questions and notes on how to improve/revise the current work
