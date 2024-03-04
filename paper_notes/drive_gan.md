# [DriveGAN: Towards a Controllable High-Quality Neural Simulation](https://arxiv.org/abs/2104.15060)

_March 2024_

tl;dr: A neural simulator with disentangled latent space, based on GAN encoder and RNN-style dynamics model.

#### Overall impression
DriveGAN uses a VAE to map pixels into a latent space. GAN-style adversarial training is used to train the VAE, thus the name driveGAN.

The proposed architecture is a very general one for a world model, actually very similar to more recent works such as [GAIA-1](gaia_1.md) and [Genie](genie.md). The original World Model by Schmidhuber is also based on VAE and RNN. Over the years, the encoder/decoder can evolved from VAE + GAN to VQ-VAE + diffusion model, and the dynamics model evolved from RNN to Transformer-based GPT-like next-token prediction. It is interesting to see how new techniques shine in the relatively old (although only two-year old) framework. Two advances: more powerful and scalable modules and much much more data.

The main innovation of the paper is the disentanglement of latent space representation into spatial-agnostic **theme** and spatial-aware **contents** in encoding stage, and further disentange **content** action-**dependent** and action-**independent** in dynamics engine.

The controllability of driveGAN is achieved via careful architecture design. More modern and scalable approach relies on scalability and more unified interfaces (e.g. natural langauge).

Action of the agent is recovered by training another model, and then used to reproduce the scene. This is similar to the idea of [VPT](vpt.md). In a way, it verifies the controllability  and geometric consistency of the simulation.

The paper has very nice discussion regarding what a neural simulator should look like. First, generation has to be realistic, and second, they need to be failthful to the action sequence used to produce them.

#### Key ideas
- Controllability via disentagnled latent representation
	- 1152 dimension latent space
	- Theme: weather, background color
	- Content: spatial info
		- Action depdendent: layout
		- Action indepdendent: object types
- Latent representation
	- Uses VAE and GAN to learn a latent space of images, enabling high-resolution and fidelity in frame synthesis conditioned on agent actions. GAN comes into play in decoding stage. --> This is upgraded to diffusion models in more recent works.
	- The **disentanglement** is achieved by forcing theme latent to be one single vector without spatial dimension.
- Dynamics engine based on RNN
	- The disentanglement is achieved by using two RNNs, forcing one conditioned on action and the other not. 
	- The neural network seems to learn meaningful representation well aligned with the intention of the architecture design. SGD will cheat and use shortcuts whenever possible.
- Multistage Training
	- Encoder/decoder training
	- Dynamics engine training with encoder/decoder frozen.
- Differentiable simulation: varying the disentangled latent vectors to reproduce or generate new contents in a controllable way
- Eval
	- FVD
	- **Action consistency**: action of the agent can be predicted by feeding two images from real videos into a dedicated model. This model can be deployed on simulated images (generated with GT actions) to verify action consistency.

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work
