# [GPT-4V(ision) for Robotics: Multimodal Task Planning from Human Demonstration](https://arxiv.org/abs/2311.12015)

_December 2023_

tl;dr: Robot operation from human demo in a zero-shot manner. Integrating observation of human actions to facilitate robotic manipulation. First work to leverate 3rd person videos into robotic demo.

#### Overall impression
GPT4v_robotics extracts affordance insights from human action videos for grounded robotic manipulation. This is performed by leveraging spatialtemporal grounding with open vocab detector [DETIC](detic.md) and focusing on hand-object relationship (grasp, release, etc).

GPT4V-robotics uses general-purpose off-the-shlef language models. It is highly flexible and can be adapted to various HW config via prompt engineering, and can benefit from the improvement of the LLM field.

Integrating vision into task planning opens up possibility of developing task planners based on multimodal human instructions (+human demo video, etc).

TAMP (task and motion planning) framework incorporates two parts. Off-the-shelf LLM decomposes human instructions into high-level subgoals. Pretrinaed skills (via IL or RL) achieves the subgoals (atomic skills).

Affordance the concept derives from the literature in psychology and cognitive science. It refers tot he potential for action that objects or situation in an environment provide to an individual (see [explanation of Gibson concept in Zhihu](https://zhuanlan.zhihu.com/p/85578008)). In robotics, it focuses on **executable actions and where such actions are possible** (cf [Saycan](saycan.md), [voxposer](voxposer.md)).

How the affordance information is leveraged to endow or decorate the task plan is still unclear. --> May need to read earlier works from the same authors.

#### Key ideas
- System architecture
	- Input: 1) human demo video and 2) text instructions. 
	- Output: affordance info and task plan, in a hardware-independent exe file saved in JSON.
- Two pipelines to analyze human demo videos, Symbolic task planner and Affordance analyzer
- Symbolic task planner
	- [GPT4V] Video analysis: transcribe video frames at regular intervals into text instructions in a human-to-human style ("Please throw away this empty can")
	- [GPT4V] Scene analysis: environment info (objects, graspable properties, spatial relationship). Effective selection of objects of interest and ignoring irrelevant objects.
	- [GPT4] Task planning: From instructions and environment (both based on texts), gives a list of tasks (Move hand, grasp, pick up, etc).
- Affordance analyzer
	- Extract info necessary for effective execution of the tasks. The tasks plans are then endowed with such affordance information.
	- Approaches to objects, grasp types, collision avoiding waypoints, upper limb posture.
	- Open-vocabulary object detector (DETIC) is used to ground (locate) object names detected by GPT4V in the RGB image.
	- Focusing on relationship between hand and object enables detection of timing and location of the occurance of grasping/releasing.
	- For each type of atomic task (move hand, rotate, etc), waypoints for collision avoidance, grasp types and upper limb posture are also extracted and serve as constraints for computing inverse kinematics of multiple DoF arms.

#### Technical details
- Summary of technical details

#### Notes
- Summary generate by ChatGPT

> this is like teaching a robot to cook by showing it cooking videos. First, the robot watches the videos and uses GPT-4V to understand what's happening - like chopping onions or stirring a pot. Then, using GPT-4, it plans how to do these tasks itself. It pays special attention to how hands interact with objects, like how to hold a knife or a spoon. This way, the robot learns to cook various dishes just by watching videos, without needing someone to teach it each step directly.