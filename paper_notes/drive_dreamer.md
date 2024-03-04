# [DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving](https://arxiv.org/abs/2309.09777)

_March 2024_

tl;dr: World model for autonomous driving, conditioned on structured traffic constraits.

#### Overall impression
First real-world world model, contemporary with [GAIA-1](gaia_1.md). Yet the controllability of the dynamics is different.

Typically the controllability of world model is only quantitative as it is hard to do (close to) pixel accurate generation with difussion models. DriveDreamer alleviates this problem and reaches near pixel accurate control with structured traffic constraints (vectorized wireframes of perception results, or `perception vectors` for short). This inspiration may be taken from [Align Your Latents](video_ldm.md).

The model takes in video, text, action and perception vectors, and rolls out videos and actions. It can be seen as a world model as the video generation is conditioned on action.

The dynamics of the world model is actually controlled by a simplistic RNN model, the ActionFormer, in the latent space of the `perception vectors`. This is quite different from  GAIA-1 and Genie where the dynamics are learned via compressing large amounts of video data.

The model is mainly focused on single cam scenarios, but the authors demo'ed in the appendix that it can be easily expanded to multicam scenario. --> The first solid multicam work is [Drive WM (Drive into the Future)](drive_wm.md).

#### Key ideas
- Training is multi-stage. --> Seems that this is the norm for all world models, like GAIA-1.
	- Stage 1: AutoDM (Autonomous driving diffusion model)
		- Train image diffusion model
		- Then train video diffusion model
		- Text conditioning via cross attention
	- Stage 2: Add action condition (interaction) and action prediction.
		- **ActionFormer** is an RNN (GRU) that autoregressively predicts  future road structural features in the latent space. **ActionFormer models the dynamics of the world model**, but in a vectorized wireframe representation.
- Eval 
	- Image/Video quality: FID and FVD
	- Perception boosting: mAP of model trained on a mixture of real and virtual data.
	- Open loop planning: not very useful.

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work
