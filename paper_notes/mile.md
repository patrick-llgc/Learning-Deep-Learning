# [MILE: Model-Based Imitation Learning for Urban Driving](https://arxiv.org/abs/2210.07729)

_July 2023_

tl;dr: End-to-end methods that jointly learns a model of the world and a policy for AD.

#### Overall impression
MILE looks quite similar to [Gato](gato.md) in the sense that they take in the observation and directly predict action. Both are based on immitation learning, with dataset generated from an RL-expert in an virtual environment. MILE does one extra step of prediction the evoluton of the environment, instead of getting it fromt he simulation engine like Gato (this is the future direction of Gato v2, as described in the Gato paper).

**MILE uses 3D geometry as an inductive bias and learns a highly compact latent space directly from raw videos.** The evolution of the environment dynamics are reasoned and predicted in this highly compact latent space. This learned latent state is the input to driving policy (output control signal) and can be decoded to BEV segmentation for visualziation and supervision.

Trajectory forecasting **explicitly** estimates the future traj of dynamic agents given past trajectory and scene context (HD Map, etc). World models build a latent representation that explains the observagtions from the sensory input of ego and its action. The future trajectory is **implicitly** encoded in the latent variable.

Critique: 

- I find the formulatio of action and state is a bit weird. First of all, action prediction should be stochastic. Second, action prediction a_t should be conditioned with the previous state s_t-1, not the current state s_t. So as I see it, o_hat, y_hat and a_hat are all there just for the auxiliary supervision.
- MILE is NOT based on LLM. Maybe combining MILE with LLM is one promising future direction, by tapping into the pool of common sense present in LLM. 
- The manuscript is quite math-heavy, and thus not super reader-friendly. The overly complicated math notations seems a bit unnecessary. 

#### Key ideas
- Probabilistic Generative model 
	- A latent variable s to model the temporal dynamics of the environment and action. **s is one step roll-out of jointly predicted future of ego, other agents and static environment.** The evolution of s is essentially the world model. Therefore action can be estimated from this predicted state. (a_t = pi(s_t, h_t))
	- A variable h to model the deterministic history (--> this can be seen as the memory bank).
	- The evolution involves a two step process
		- a determinstic update of $h_{t+1} = f_{\theta}(h_t, s_t)$
		- a stochastic update of $s_{t+1} \tilde N(\mu, \sigma)$. s is **sampled** from a normal distribution. Both $\mu$ and $\sigma$ is function of $h_{t+1}$ and $a_t = \pi_{\theta}(h_t, s_t)$. Note that the action is inferenced, without access to the GT action. The inference from s to a is deterministic, which can be understood as that a is decoded from s. 
	- from a different persepctive, the evolution is 
		- action prediction
		- next state prediction
- Inference network
	- Observation encoder: x_t 512-d.
		- Image feature: LSS to 64x48x48, mapped to 480-d, about 300x compression. 
		- Then route map feature and speed feature, both 16-d, are concated with image featrue. 
	- Posterior network to estimate the mean and variance of s (512-d). 
	- The inference model is there just for the KL div matching.
- Generative network
	- gated recurrent cell $f_\theta$, to update the memory bank.
	- policy network to estimate action from s_t-1 (last state) and h_t-1 (past history). 
	- prior network to estimate the mean and variable of s. 
	- optionally, the BEV seg decoder from s. 
- Driving in imagination
	- s1, h1 --> a1 (decode last action) (--> Not quite a prediction, but rather a decoding)
	- s1, h1 --> h2 (update memory bank)
	- a1, h2 --> u, sigma --> s2 (estimate new joint future state)
	- Essentially the model rolls out the latent variable encoding the future, one step at a time. This step can be iterated to generate longer sequence of futures in latent space. The predicted future can be decoded and visualized.
	- The ablation study shows that we can drop 40% of input within a 2 second window, but still drives good. --> This would be better benchmark against a simple baseline with constant velocity of the last timestamp with observation. Maybe this simple kinematic baseline will do.
- Training data: 
	- 32 hours of expert RL agent driving in CARLA, at 25 Hz. 
	- Single front camera, 2.9 M frames. 
- Loss function
	- Image reconstruction: L2 loss
	- BEV segmetnation: CE loss
	- Action: L1 loss
	- KL divergence between posterior and prior distribution of s. 
		- The KL divergence matching framework ensures the model predicts actions and future states that explains observed data. 

#### Technical details
- Both the compact BEV representating (x) and the latent variable to describe the dynamics of the world (s) are 512-dimention. 
- Fully recurrent inference does not lead to performance drop. --> This seems to be the standard process of any RNN-based network though. 

#### Notes
- The ability to estimate scene geoemtry and dynamcis is paramout to generating complex and adaptable movements. The accumulated knolwedge of the world (common sense) allows us to navigate effectively.
- MILE is an IL method, and does not need interaction with the environment. Part of the ability is through hallucination of the future, or driving by imagination. The trianing entirely based on offline driving corpus offer strong potential for real-world application.
- DAgger proposes iterative dataset aggregation to collect data from trajectories that are likely to be encountered by the policy during deployment. It solve the problem of covariate shift. 
- Learning latent dynamcis of a world model from image observations was first introduced in video prediction.
- The ego action does indeed affects the environment. MILE is action-conditioned, allows how other agents respond to ego-action. Previous works often assume that the environment is not affected by ego. This is too strong an assumption. ("world on rails"). 
	- "To support learning from pre-recorded logs, we assume that the world is on rails, meaning neither the agent nor its actions influence the environment." -- from "Learning to drive from a world on rails", ICCV 2021
- Driving score = Route completion (recall) X infraction penalty (precision). The penalty is a cummulative score. The driving score is too coarse. Better use the cumulative reward, which measures the performance at timestep level.
- Q: is dynamic or static environment more challenging in autonomous driving? 
	- Initially it is dynamic objects, given that we can use HD map information.
	- Then static environment is more important to predict a local HD map. This is to reduce dependency on HD Map to ensure the generalization and accessibility of AD.
	- Then the dynamic objects will be more important, especially in the apects of prediction and planning.
- Q: is it better to seperate static and dynamic scenes in the latent variable s?

## TODO: to ask the author
- Q: why the prior does not have access to the ground truth action? 
- Q: why action is not conditioned on previous state? 
- Q: why is the driving policy pi deterministic? So $\pi$ is better viewed as a decoder, but not a policy rollout. 
- Q: P8, the paper touched upon probabilistic modeling, indicating that a deterministic driving policy cannot model multimodality. This all makes sense (although I think MILE's claimed "policy", actually a decoder, is deterministic). In order to prove this point, maybe an ablation study with sigma set to zero should be carried out. However the paper cited an ablation study of removing the KL divergence matching.

## Questions
- What is conditional immitation learning? (CIL)
- What is on-policy and off-policy?
- What is offline RL?

## To read
- Unsupervised Learning, nature 1989
- Recurrent world models facilitate policy evolution, neurIPS 2018
- Learning latent dynamics for planning from pixels, ICML 2019 (Recurrent State-Space Model, RSSM)
- Mastering atari with discrete world models, ICLR 2021
- Learning from all vehicles
- 2016 nvidia end to end driving
- Exploring the limiation of behavior cloing in autonomous driving, 2019 CVPR
- PRECOG: a la scene-transformer?
- Video prediction
	- Stochastic variational video prediction
	- Stochastic video generation with learned prior
	- Stochastiv latent residual video prediction