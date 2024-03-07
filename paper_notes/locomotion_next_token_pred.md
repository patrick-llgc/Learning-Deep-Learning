# [Humanoid Locomotion as Next Token Prediction](https://arxiv.org/abs/2402.19469)

_March 2024_

tl;dr: A motion controller based on next token prediction of sensorimotor tokens for bipedal humanoid locomotion.

#### Overall impression
The paper tackles humanoid control, specifically humanoid locomotion (standing upright and moving legs) as an e2e control problem. The sequence of sensory observations and motor actions makes up **sensorimotor trajectories**, as the sentence of the physical world. Note that there is NO images or perception involved, but only streams of relatively sparse, structured output.

This paper, alongside RT1, signifies a new era of Big Data through Transformers as a Control Policy. Model based control --> DRL (too fragile to OOD, complex corriculum) --> Transformers

A causal transformer is trained with immitation learning via autoregressive prediction of sensorimotor trajectories. The input to the autoregressive model is a pair of tokens of observation (joint encoders, IMUs) + action (motor commands).

Robotics data is diff from language as robotics data are naturally multimodal and high dimensional. In order to scale up training, it is inevitable to deal with missing modality during training.

It rolls out observation and action jointly, and in this way, it is a world model of sensorimotor input. **The state-action prediction yields better results than training with action-only prediction.** The joint pred task forces the model to learn richer representation of the world that are beneficial for action prediction. --> This is why we need World Model.

> As the base models get exponentially more expensive to train, all researchers (no matter what institution you are at) will face the same engineering constraint: there is only enough resources to train the biggest model once. All post-training capabilities need to be derived from that base model, and because it’s hard to anticipate what the downstream tasks look like, you must prepare the base model for all possible tasks. In other words, your foundation model’s training objective should be for the full generative model of the data, such as an autoregressive next-token predictor (e.g. GPT) or a diffusion process (e.g. a video generative model like Sora), or some combination of the two. If you **throw your base model budget on a conditional density modeling problem**, e.g. “predict all the robot actions from the video”, it might not be a good base model for many tasks that you might care about later. This only becomes more true as the cost of the base model grows. ---- [All Roads Leads to Robots](https://evjang.com/2024/03/03/all-roads-robots.html)

The model can transfer to real world when trained with ONLY 27 hours of data. Another interesting fact is that the transformer based policy is smoother and more accurate than the RL policy, although the model is trained with trajectories produced by this RL policy. (青出于蓝? Why?)

#### Key ideas
- Data source: diverse dataset with potentially missing modalities. Scraped from the internet and simulators.
	- Prior RL policies (o + a)
		- 10s x 10k, RL in Issac Gym.
		- Data generation policy: The trajectory are conditioned on velocity commands with high variety.
	- Model based controllers (o + a)
		- 10s x 10k x 2
		- Data generation policy: Default cofnig for one and randomized config (leg, clearance, floor properties) for the other.
		- Diverse velocity profile
		- Generates torque, not consistent with action space, so dropped out actions.
	- Motion capture (o only)
		- MoCap capture human keypoints in 3D, and inverse kinematics are used to find corresponding robot pose.
	- Youtube videos (o only, much noisier)
		- Reconstruct human videos by using CV techniques and retarget both motion capture and youtube trajectories via inverse kinematics. 
- HW: Agility Robotics
	- 30 DoF, 20 actuated
	- Hard to optimize fast, so learning based approach is favored.
- Training: the general transformer model autoregressively predict shifted input seq. 
	- L2 loss used during next token prediction. --> This is already good enough per the author. This is a bit questionable.
	- CE on quantized token not used. 
	- Hidden state of 192-dim.
- Tokenization: one token contains a (observation + action) pair.
- Missing modality: treated as trajectories with action masked. 
	- A mask token [M] to replace action.
	- [M] randomly init and learned with the model.
	- Loss ignored for [M].
- Inference
	- Always have (o, a) pairs available, predict next (o, a) pair. Discard o, and keep a.

#### Technical details
- Relative small transformer that can be deployed on CPU (Agility perhaps don't have a GPU onboard).
- In a classic robot, the controller is the brain and generate motor commands to various motors and adjust the commands based on the feedbacks from sensors.
- The prediction is in a modality aligned way to better align training and inference data distribution. In other words, during inference, action prediction needs to be conditioned on predicted observation, and this corresponds to modality alignment during training.
- High DoF robots spells challenges for classic optimziation based appraoches. --> reasons why we need learning.
	- Increased complexity of kinematic and dynamic models. Makes it harder to have accurate models.
	- Computational Load increases significantly, making it challenging to meet real-time requirements
	- Local Minima: easy to get stuck in local minima for high dim optimization
- There are two kinds of commands mentioned in the paper. 
	- Walking command provides high level goals, by specifies the direction and speed at which the robot should move. It includes linear velocity (heading command) and its angular velocity (yaw command).
	- Motor commands are the detailed instructions that execute this goal by controlling the robot's mechanical parts

#### Questions
- [A nice review on Youtube](https://www.youtube.com/watch?v=LLprTlQsIJc&t=5445s&ab_channel=hu-po)
- "Prompting our controller with negative values for the heading command": This is the high level command for the entire robotics. How does the high-level goals of walking commands affect the autoregressive model? In other words, how is the autoregressive prediction conditioned on the high level goals?
	- Via correspondence with the author, controller prompt is the operator command, and NOT actions. **Operator commands are inside observation** (not described in the paper), used to control high-level behaviors of the robot, such as walking speed or steering speed.
- Why still continuous tokens, not discrete tokens? Or can it even be called tokens?
- How large is the youtube dataset?
- What is the freq in Hz is the sensorimotor trajectory?


