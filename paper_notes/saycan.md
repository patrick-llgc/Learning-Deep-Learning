# [SayCan: Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691)

_July 2023_

tl;dr: Generate natural language actions that are both feasible and contextually appropriate. Combine generated action and the affordance to pick the most probable action. Overall score = say * can.

#### Overall impression
SayCan's key idea is to ground language models through affordance functions. The affordance functions capture the log likelihood that a particular skill will succeed in a particular state. 

Critiques:

* how the actions are picked depends on how the actions/skills are phrased.
* Not end-to-end. The image conditioning only comes at the second stage during the value function. A better formulation maybe using a VLM (like [Flamingo](flamingo.md) directly) in the first phrase.
* there is no consideration of the success status of the low level policy. Assumes the action will succeed always. Does not consider how to recover (pick up an apple after picking it up again).

Advantages:

* LLM are frozen, and can be updated independently
* Value function for each skill are updated and shipped independently.
* Not end-to-end, but very modular. Modularity is the key.

This is a quite big project, with many authors contributing to many aspect of this complex engineering project. 



#### Key ideas
- The robot is equipped with a repertoire of learned skills (policy bank) for basic "atomic" behaviors and these skills are capable of low-level perception and control. --> This is improved in [VoxPoser](voxposer.md).
- Part 1 Say: The model will get the likelihood of an atomic action (the contextual properness), whether it will help progress toward completion of the task, describing how useful it is. This is purely based on the LLM, NOT conditioned on the image.
- Part 2 Can: a value function tells how likely the action will succeed given the current observation. For example, if there is no apple in the image, then "pick up an apple" would not be scored high.
- Find the skill with best score=say*can.
- Execute the action, update the state, then repeat. 

#### Technical details
- LLM may generate a reasonable narrative, but may not be applicable to a particular agent in a particular environment. 
- SayCan needs a set of skills, each of which has a policy, a value function and a short language description. 
- 551 skills in total.
- HW: Everyday robotics, with 7 DoF arms and two-fingered gripper. Open-sourced virtual version uses UR5 ([universal robots](https://www.universal-robots.com/products/ur5-robot/)) robot and uses CLIPort to output pick-and-place location.
- Better LLM leads to better performance in robotics.

#### Notes
- youtube video by Yannic](https://www.youtube.com/watch?v=Ru23eWAQ6_E&t=40)