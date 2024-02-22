# [End-to-end Autonomous Driving: Challenges and Frontiers](https://arxiv.org/abs/2306.16927)

_February 2024_

tl;dr: Very good high level overview of end-to-end autonomous driving, with a focus on planning. 

#### Overall impression
This can be seen as an extensive background review for [UniAD](uniad.md). The review part for IL and RL, and closed-loop evaluation part is well-written. 

End-to-end systems (or referred to as visuomotor system, i.e. vision in, motor control out) contrasts with modular pipeline involving rule-based design. E2E systems as fully diff programs that take raw sensor data as iput and produce a plan or low-level control action as output. 

E2E does **not necessarily** mean one black box with only planning/control outputs. Yet the E2E system mentioned in this paper is mainly still a modular based design, but every component and the entire compound system is differentiable. Actually there is another E2E architype which completely eliminates the notion of perception, prediction and planning, and it is gaining popularity in embodied AI (robotics) field.

Conventional CV tasks are typically dense prediction task (obj det, semantic seg). Yet autonomous driving predicts very sparse signals. (So does the computer-aided diagnosis for medical imaging systems.) **This means the sparse signal alone cannot guarantee good representation learning. This does not necessarily exclude the possibility of a black-box design.** 


#### Key ideas
- Why E2E
	- Proxy KPIs (optimization goals across modules) are different.
	- Errors will be compound across modules.
- IL and RL and policy distillation
	- BC (IL): supervised learning. 
		- Good: simplicity and efficienty.
		- Bad: covariate shift, DAgger to address
		- Bad: causal confusion
	- IOC (IRL, still IL). GAIL (generative adversarial IL) designs the reward function as an adversarial objective to distinguish the expert and learned policies. 
		- Cost volume and a kinematically feasible trajectory set. 
		- Max-margin loss is used to encourage the GT to have the lowest cost and others have high cost. See [NMP](nmp.md).
	- RL: learning by trial and error. Not converged to a specific RL algo yet. 
		- Good: Superior performance when combined with IL. See [BC SAC](bc_sac.md).
		- Good: can be used to perform policy distillation. Train a teacher model with access to privileged info. [LBC](lbc.md).
		- Bad: req environ to allow unsafe actions in exploration phase. Needs simulation.
		- Bad: sparse and simplistic rewards (progress, collision avoidance) encourages risky behaviors.
- Evaluation
	- Interesting safety-critical cases occur rarely but are diverse in nature. Efficient generation of realistic safety critical scenarios that cover the long tailed distribution remains a significant challenges.
	- Offline (open loop): against prerecorded expert driving trajectories. 
		- Good: This does NOT require a simulator.
		- Bad: test-time behavior may deviate from expert driving corridor, and it is impossible to evaluate if the system can revover from such drift.
		- Bad: cannot evaluate multi-modal behavior. 
	- Online (closed loop): involves constructing a close-mimicking simulation environment. It has 3 main subtasks: parameter init, traffic simulation and sensor simu. All 3 components have a rule-based system variant, and an evolving data-driven approach.
		- Parameter initialization: Procedural generation, with probabilistic sampling for a set of predefined parameters. The parameterization of the environment is handcrafted. The data-driven approach may not cover extreme corner cases, which is critical for test coverage. **Procedure is still needed.**
		- traffic simulation: rule-based traffic simulators, i.e. IDM (intelligent driver model), simplistic and inadequate in urban environ. The data-driven approach is NPC AI. **NPC AI for the win.**
		- sensor simulation: graphics based approach vs data driven (Nerf or 3DGS). **Data driven will win.**
- Foundation models and visual pretraining
	- **LLM for Autonomous driving remain unclear, except VLN (vision language navigation).** A feasible solulton is to train a video predictor that can forecast long term predicion of the environment in 2D or 3D (better in latent space per LeCun).
	- Two stages: encoding state space into a latent feat representation, and then decoding the driving policy with intermediate features. 
	- Representation/pretrain visual encodeer with proxy pretraining tasks.
	- Human-defined pretraining tasks (e.g. segmentation) often imply info bottleneck in the learned representation, and redudance info unrelated to driving may be included. (recall and precision both imperfect)
	- MTL (multitask learning): semantic segmentation and depth prediction are typically used as auxiliary task to learn a good representation from images.
- World model and MBRL (model based RL)
	- RL suffers from large sample space and low sample efficiency. World model comes to the rescue by reducing sampling complexity.
	- MBRL explicitly model the world model, consisting of transition dynamics and reward functions, and the agent can interact with it with low cost.
	- Learning world model in raw image space is not suitable for autonomous driving, as small important details (TFL) could be easily missed.
- Policy distillation
	- Train a privleged agent with RL, and then use it to provide supervision at the output planning level, and/or the feature level.
- Causal confusion
	- Temporal smoothness makes past motion a reliable predictor of next action. Models trained with multiple frames can become overly reliant on this shortcut. 
	- Possible solutions: One could use only single frame image to predict steering. Or random input dropout of ego speed profile. Or to train an adversarial network to eliminate its own past from its immediate layers. Or to update keyframes where a decision change occurs. 
	- The **most promising solution** seems to use a BEV representation and construst a local map with other agents histories. This was motivated in lidar (MP3, NMP, LAV). 
	- Network may cheat with low-dim features. For example, it may use ego speed history to cheat ("I am braking becauase my speed is low"). Network may use the behaviors of surrounding agent while ignore the TFL. (See [response to Tesla FSD V12 test drive](https://www.zhihu.com/question/619544346/answer/3188081298)).
- Covariate shift and data engine
	- Reasons: 1) deployment in unseen environ, 2) reactions from other agents diff from training time. 
	- DAgger: current trained policy is roled out in each iteration to collect new data. This is essentially the **data close-loop or data engine** required for e2e autonomous driving.
	- Downside of DAgger: requires expert to query online all the time
	- Domain adaptation is mainly focused on sim-to-real. Geography-to-geography and weather-to-weather adaptation are handled through dataset scaling.
	- The design goal of data engine: it should support mining corner cases, scene generation and edtiting to failiate data driven evlauaton. 

#### Technical details
- Multimodality: this refers to the multiple possible outcome, instead of the multimodal input.


#### Notes
- More resouces can be found in [the accompanying github page](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving)
- Papers to explore further
	- Early E2E endeavors: ALVINN, DAVE-2 (2016, NVidia)
	- closed loop training: LBC (policy distillation), WoR, TCP
	- Roach: RL agent to generate data to train IL
	- VLN (vison language navigation): LMNav
	- World models: Dreamer series, ISO-dream, [MILE](mile.md)
	- DAgger: LBC and DARB
	- Representation learning: PPGeo, ACO
	- NPC AI: BITS, CTG

#### Question
- FSD causal confusion?