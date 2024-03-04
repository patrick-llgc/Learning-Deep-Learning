# [Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos](https://arxiv.org/abs/2206.11795)

_March 2024_

tl;dr: Enables agents to learn to act by observing unlabeled videos from the internet

#### Overall impression
It is valuable to effectively leverage unlabeled video (no actions recorded at each frame) for sequantial decision training (robotics, AD, etc). IL is the simplest when demo are labeled with corresponding actinos. VPT and [Genie](genie.md) adopts quite different approaches. Genie learns a latent action model jointly with dynamics model, while VPT uses a noncausal IDM to pseudolabel data. In comparison, VPT seems to work well with a specific domain, but Genie seems to learn a more general representation of action space and have better cross-domain generalization.

VPT leverages internet-scale unlabeled data for training agents in sequential decision-making domains. By combining semi-supervised imitation learning with a small amount of labeled data to train an **inverse dynamics model (IDM)**, agents can perform complex tasks in Minecraft, achieving near or at human-level performance in some cases. 

Given a fixed annotaiton budget, what is the most efficient way to spend the money? The answer provided by VPT is to train a non-causal Autolabel system and then use it to pseudolabel unlabled data. This is widely used in engineering practices such as autonomous driving. 2k hours of data only cost $2000 or lower to collect and unlock the full potential of massive unlabeled online data for use in BC.

It is MUCH (2 orders of magnitude) easier to predict an action with future information in a non-causal fashion (IDM), or with hindsight information, than do it causally (VPT). This power of hindsight can be heavily leveraged in autolabeling.

The use of a IDM model to predict action is very much like that in [DriveGAN](drive_gan.md) for action consistency analysis. 

This is NOT a world model (as it does not predict how the world evolves), but a giant end-to-end IL agent. It is the first to report non-zero success rates on crafting a diamond pickaxe, very long into the technology tree (24k action steps later).

For a foundation model for AD, action needs to be trained in pretraining stage to incorporate knowldge, or to learn the action priors. Another way is to come up with an architecture with flexible input format that can deal with missing modes (some labled data and some unlabeled data).

Concurrently there is the best paper of NeurIPS MineDojo, which leverages advances in LLM and RAG to develop an agent. These two studies are complimentary, and VPT is much closer to a production system (浓浓的工业风美感).


#### Key ideas
- Data 
	- [VPT](vpt.md) crawled 270k hours of Minecraft gameplay data and filtered into 70k hours. 128x128 resolution. --> [Genie](genie.md) crawled 244k hours of platform gameplays and filtered into 30k hours, about half of the size of VPT. 160x90 resolution.
	- A small set of labeled data (2k hours) is used to train an IDM. Ablation study shows that as few as 100 hours would be sufficient!
	- IDM labels a large corpus of unlabeled video data (70k hours, or roughly 8 years), enabling the training of a general behavioral prior.
- VPT Foundation model: **Pretraining** stage of ChatGPT.
	- Frames in, Actions out. 
	- 0.5 B parameters, same architecture for both VPT and IDM models. The only diff is VPT is causal and IDM is noncausal and can look into the future.
	- VPT provides "behavioral priors", the model acts like a human when it does not know what to do (not value aligned yet).
- Finetuning
	- Can be IL findtuned on specific dataset. This is essentially **SFT** stage of ChatGPT dev. For example, building a house specifically by contractor ("老司机数据"), then some skills will improve by 2 orders of magnitude. 
	- Can also be finetuned by RL. This is like RLHF, closed loop training in ChatGPT. Note that the simulator of minecraft is essentially the game, which is already there for closed loop training. For AD, we have to have a neural simulator.
	- RL finetune on top of SFT will get diamond pickaxe. RL finetune on top of foundation model along, without SFT will not.
- Scaling law
	- VPT also exhibits nice scaling property.
	- The hypothesis of VPT is that good perf can be achieved with much simpler method, given enough data.

#### Technical details
- RL from a strong IL baseline will improve performance, but RL from scratch will get you nothing. Life is also like this.
- If you can train two models separately, do not train them jointly. Train separately then finetuned jointly will lead to better results. This is the key insight enabling VPT to perf much better than previous work which also used the concept of IDM.
- Training cost
	- IDM: 4 days, 32 A100 GPUs.
	- VPT: 9 days, 720 V100 GPUs.
- Action space: discrete
	- Binary keystrokes: 20 kinds
	- Mouse movements: 11 **foveated** binning (more fine-grained for smaller movements and more coarse for larger movements) along x and y axis.
- KL loss to bind with original model to curb catestrophic forgetting.
- Natural language conditioning: natural language narration is used to condition the agent behavior, by adding a NLP layer and finetune VPT. The model shows some degree of steerability (controllability) but this direction needs more work.

#### Notes
- [Yannic's review on Youtube](https://www.youtube.com/watch?v=oz5yZc9ULAc)
- [Twitter review](https://twitter.com/jeffclune/status/1540002278366621696?s=20)