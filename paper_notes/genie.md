# [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)

_March 2024_

tl;dr: A 11B world model trained in an unsupervised manner from unlabeled internet videos.

#### Overall impression
The main innovation is the training of a latent action model (LAM) that is trained via unsupervised learning. Basically it does discrete clustering of continuous actions. Action conditioned data is hard to obtain. Genie unlocks the potential of using almost limitless online videos for training a world model.

The tech report is very concisely written, as other reports from DeepMind, and tons of useful details for training in the appendix.

The model differs from [GAIA-1](gaia_1.md) in that GAIA-1 still uses video data with action and text annotation. Architecture-wise, GAIA-1 uses a dedicated video decoder based on diffusion model, but Genie uses the decoder of the tokenizer. --> Maybe this can explain the poor image quality.

A world model enables next-frame prediction that is conditioned on action inputs. 
Genie is a foundation world model, and can be used for training generalist agents without direct environment experience at agent training time.

#### Key ideas
- Architecture
	- Video tokenizer
	- Latent Action Model (LAM)
	- Dynamic Model
- Spatiotemporal video tokenizer
	- VQ-VAE, MaskViT. 1024 unique codes in code book.
	- Memory efficient ST-transformer, with interleaved spatial and temporal attention. In a way it is factorized attention.
	- ST-transformer is more efficient, scales linearly with num of frames. 
	- ST-transformer is less prone to overfitting (and thus higher perf) compared with the full blown spatialtempral attention.
- Latent Action Model (LAM)
	- 8 unique codes in code book.
	- Can infer latent action between each pair of frames
	- VQ-VAE, to map continuous actions to a small discrete set of codes. 
	- At inference time, only the VQ code book is retained in inference time, and the entire LAM is discarded.
- Dynamics model
	- Takes in video tokens and latent actions, and **autoregressively** predicts net frame using MaskGIT.
	- The transformation between pixels and latent vectors is through the encoder and decoder of the video tokenizer.
	- Cross-entropy loss.
	- *Action condition via additive embedding* (cf. concatenation in previous works).
- Training
	- First train video tokenizer
	- Co-train latent action model (directly from pixels), and dynamic models (on video tokens)
	- 942B token during training. 90x ratio to model size.
- Eval
	- Video quality via FVD (Fréchet Video Distance)
	- Controllability via a diff PSNR. A higher score indicates the diversity of generated next frame with diff action, thus higher controllability.
- Generalization
	- Generates well to OOD image prompts.
- Robotics
	- A smaller 2.5B model is trained on [RT1](rt1.md) robotics data. It learn a consistent action space. Same latent action generates similar effects for different input, such as putting down, move left. --> This means **the compression in action space leads to high level semantic meanings**. One latent action corresponds to a complex combination of actions. So the latent action space codebook size could be smaller than the total DoF of the robot.
	- This presents a path to using larger video datasets to create a foundation world model for robotics.
- Agent training: this in another sense proves the consistency and meaningfulness of the latent action space
	- First collect expert demo
	- A frozen LAM model first labels an expert demo video
	- The discrete latent action space needs to be mapped to real actions, defined by the native DoF of robotic actions.
	- An agent is trained to predict the next possible latent action by the expert
	- Then latent action mapped to real action.

- Scaling law

#### Technical details
- GAIA-1 and Genie have similar model sizes. GAIA-1 has a total of about 10B (0.3B image tokenizer + 6.5B world model + 2.6B video diffusion model). Genie has a total of about 11B (0.2B image tokenizer + 10.7B dynamic model).
- Why train LAM on pixels, instead of tokens?
	- Action model trained on video tokens loses generalization capability, compared with training directly on pixels.
	- Some info about video dynamics and movements night have been lost during tokenization.
- The mapping between latent action and real action
	- Unclear at first, how each latent action will impact next frame gen.
	- But action remained **consistent** across diff inputs, making it similar experience to learning the buttons on a new controller. 

#### Notes
- Questions and notes on how to improve/revise the current work
