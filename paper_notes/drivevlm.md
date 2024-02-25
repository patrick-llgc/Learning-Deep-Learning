# [DriveVLM: The convergence of Autonomous Driving and Large Vision-Language Models](https://arxiv.org/abs/2402.12289)

_February 2024_

tl;dr: A hybrid system to leverage VLM for long-tail scenario understanding.

#### Overall impression
In a nutshell, DriveVLM maps vision observation into language space and ask a wise blind man (LLM) to perform the driving task.

Urban driving is challenging due to long-tail scenario understanding including rare objects, rare scenes and negotiation with road agents. (Past planning stack focuses on trajectory level action and neglecting the decision level interactions. This is what makes autonomous driving systems unnatural/misaligned with experienced human driver.)

DriveVLM uses VLM to tackle all these issues. It uses VLM's generalization capability to perform rare object recognition out-of-the-box, and uses LLM reasoning for intention level predition and task-level planning. The power of VLM is harnessed through a carefully designed CoT. --> In this sense, it is still a **modular design, but the system is constructed through prompting, making it more similar to modular e2e approaches such as UniAD than e2e planning**. 

DriveVLM-dual is a hybrid system that leverages the zero-shot and reasoning power of VLM, and also suplement the short commings of VLM (such as 3D grounding and long latency) by traditional AD stack.

The system seems to be more explanable than e2e planning task, as all VLM outputs can be logged for analysis and debugging.

The idea of the paper is very solid and production-oriented. It is also well-executed and validated, through a customized dataset. However the paper is not well written, with lots of important details missing for the understanding, let alone reproduction, of this work. See the Notes section. 

#### Key ideas
- VLM arch design
	- Input: 4 frames, 1 sec apart. The images are resized to 448x448 before feeding into VLM. --> Is this too small for traffic light understanding?
	- Qwen. Need to be finetuned. SFT'ed model is better than frozen ChatGPT4V.
- CoT prompting of VLM
	- Scene description
		- Environ description, such as weather, time, road, lane
		- Critical objects: not only dynamic road users, but also debries (fallen trees, etc). Semantic class and 2d bbox.
	- Scene analysis: analyzes possible intent-level behavior of the critical object
	- Hierarchical Planning
		- **Meta actions**: a short-term decision of the driving strategy. **A driving decision is a seq of meta-actions.** The actions fall into 17 categories. Each meta action is pivotal and marks different segment of the driving scenario.
		- Decision discription
		- Waypoints
- Hybrid (slow-fast dual) system with traditional stack
	- Use perception results as grounding. 3D objects from pereption are backprojected into images and match with 2D bbox observations. The 3D object's track are then passed to LLM for enhacned performance.
	- DriveVLM latency is 1.5 seconds, but good enough for providing high level guidance. The coarse and low-frequency waypoints are used for a faster refining planner (which can be optimization based or NN based).
- SUP task (scene understanding and planning): a new task
	- Dataset mining: from CLIP discription, and past driving logs
	- Annotation: waypoints is self-labeled via logs, all others are manually annotated with 3 annotators. 
	- It has a nice summary of corner case scenarios in Appendix A2.
- Eval
	- scene description: LLMs to assess similarities. Hallucinated info are penalized.
	- meta-action eval: generate semantically equivalent alternatives and then do a max common length match.

#### Technical details
- Dense cost maps is unreliable as it heavily depends on costs tailored through human experience and the trajectory sampling distribution.
- Driving caption datasets: most only focusing on perception and does not cover planning, and the scenarios are typically easy.

#### Notes
- Questions for the authors:
	- How large is the SUP dataset?
	- How are VLM models SFT'ed? How much does SFT matter?
	- Is the resized image (448x448) too small for traffic light undertanding? Any performane validation in intersection with traffic light control will be interesting. 
	- How is the coarse waypoints finetuned for DriveVLM? And how are the VLM branch and traditional branch mixed (say, VLM inference once for every 5 times of traditional stack)?
	- Page 12 A1-1: Jumbled sentence.
	- Navigation information seems also added to the prompt. This is not mentioned in the main text.
	- Figure 17: the coordinates of the two police are the same? Is this a typo or error by the LLM? Also, the images do not seem to be synchronized. This is particularly obvious in Fig. 17, but same issue happens in other figures too.
	- It is insufficient to drive the car with front cam only. So in production, it is necessary to scale up to all cameras. 