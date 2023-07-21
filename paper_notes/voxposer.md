# [VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models](https://voxposer.github.io/voxposer.pdf)

_July 2023_

tl;dr: Compose robot trajectories through a dense sequence of 6DoF end-effector waypoints. Ground LLM output to reality. It is a breakthrough in manipulation task, which uses LLM for in-the-wild cost specification.

#### Overall impression
LLM can be the key components to power embodied AI. Yet two gaps remain: How to condition or ground the input of LLM to reality, and how to ground the output of LLM to reality, to fully bridge the perception-action loop. The former challenge can be largely overcome by recent progress in VLM (vision language model) such as [Flamingo](flamingo.md) and [VisionLLM](vision_llm.md). The later challenge is tricker. VisionLLM provided some clue of how to expand LLM vocabulary to include trajectory tokens. [VoxPoser](voxposer.md) provides a brand new method to generate optimization cost function landscape instead of generate ultimate trajectory. An zeroth order optimization method is used to solve the cost function.

VoxPoser extracts language-conditioned affordances and constraints from LLMs and grounds them to the perceptual space using VLMs, using a code interface and without training of either components. (Both LLM and VLM is **frozen**). Voxposer leverages LLMs to compose the key aspects for generating robot trajectories (value maps) rather than attempting to train policies on robotic data that are often of limited amount or variability.

One major drawback of VoxPoser is the **lack of end-to-end differentiability**. It is zero-shot, and no finetuning is required. This also limit the improvement close-loop of this pipeline. If we collect some corner case data in the field, there is no clear way to perform data close-loop to improve the performance of the online algorithm.

The model can work with dynamic perturbations, but these perturbations happen much faster than the robot can move, so essentially no active prediction needs to be performed. The robot just needs to adapt to the new environment.

This work seems to be an expansion of [Code as policies](cap.md), language model generated programs (LMPs). CaP still relies on manly created 

Key questions: where does the LLM's capability to generate affordance map come from?

#### Key ideas
- Predefined motion primitives (skills) are still the de-facto method to carry out physical interactions. This remains a major bottleneck for robotics planning.
- 3 step process
	- LLM break free-form language instruction to a series of sub-tasks, each specifying a manipulation task.
	- LLM can generate python code to invoke perception APIs to obtain spatial-geometric information of relevant parts (assuming perception is a solved problem).
	- LLM then manipulate 3D voxels to prescribe rewards or cost at relevant locations in 3D space. --> This is the meat of the paper. 
	- Then the composed value (cost) map can serve as objective function for motion planners to synthesize robotic trajectories. 
- A large number of tasks can be characterized by a voxel value map V in the observation space of the robot. The cost can be approximated by accumulating the values of the "entity of interest" (such as an end-effector) traversing through V. LLM can write python programs to compose value maps to accurately reflect the task instructions. --> Where does this capability come from?
- Learn a dynamics model of the world --> what for?
	- The standard setup is that the robot interleaves between data collection and training a dynamics model (L2 between predicted observation and real observation. This seems to be the **world model**). 
	- zero-shot synthesized trajectory serves as a useful prior to explore the world efficiently. This actually speeds up the learning process. "A good enough expert for demonstration", 80% of human expert demo level.

#### Technical details
- End-effector (末端执行器): In robotics, an end-effector is the device or tool that is attached to the end of a robotic arm, which is used to interact with the objects in the environment. For example, in manufacturing, a robot might use a gripper as an end-effector to pick up and place parts on an assembly line, while in healthcare, a surgical robot might use specialized tools as its end-effector to perform complex medical procedures.
- Actuator: The end effector is the part that performs a specific task on the environment, while the actuator generates the motion needed for the end effector to complete its task. Examples of actuators include electric motors, pneumatic cylinders, and hydraulic pumps.
- Code as policy: [Code as Policies: Language Model Programs for Embodied Control](https://arxiv.org/abs/2209.07753)
- Zeroth-order Optimization (ZO): Zeroth-order optimization (a.k.a. derivative-free optimization / **black-box optimization**) does not rely on the gradient of the objective function, but instead, learns from samples of the search space. It is suitable for optimizing functions that are non-differentiable, with many local minima, or even unknown but only testable. Examples of applications of ZO include hyperparameter tuning, neural architecture search. (from [github python repo](https://github.com/eyounx/ZOOpt)) (鸟枪法？)
- LLM excel at inferring affordance and constraints, conditioned on language.
- Open vocabulary detectors: ViLD, MDETR, OWL-Vit, DETIC
- Prompting structure follows LMP.
- The system assumes availability of RGBD data. 
- Value maps: affordance (sink), avoidance (source), end-effector velocity, end-effector rotation, gripper action.
- HW: [Franka Emika Panda robot](https://www.franka.de/research/).

#### Notes
- Language is a compressed medium through which human distill and communicates their knowledge and experience of the world. LLM has emerged as a promising approach to capture this abstraction. 
- LLMs are believed to internalize genealizeable knowledge in the text form, it remains a question about how to use such generalizable knowledge to enable embodied agents to **physically act** in the real world. 
- Skills: existing approaches typically relies on a repertoire of manually designed or pre-trained motion primitives (skills). The reliance on individual skill acquisition is typically a bottleneck due to lack of large-scale robotic data. Manual design and data collection of **motion primitive** (like anchors) is laborious. --> This is not a problem for autonomous driving though. So maybe we do not need fancy highly generalizable cost composer for autonomous driving.
- It is **infeasible for LLM to directly output control actions in text**, which are typically driven by high freq control signals in high dim space. 
- Difficulty of Autonomous driving vs general robotics: general robotics are simpler in that prediction is not required for slow motion robotics. However autonomous driving has large amount of human demonstration data. 
- Robotics data are sparse, so relying on common sense, rather than huge amoutn of human demo, is the key to general AI. 

#### Questions
- Why skills is not typically a thing in autonomous driving? --> Maybe only required in the context of RL?
- Why LLM can be used to generate value maps? Where does this generalization capability come from?
