# [VideoGPT: Video Generation using VQ-VAE and Transformers](https://arxiv.org/abs/2104.10157)

_March 2024_

tl;dr: Uses VQ-VAE and Transformers to efficiently and  autoregressively generate high-fidelity videos from textual descriptions or other conditional inputs.

#### Overall impression
The paper's main contribution lies in its novel use of VQ-VAE for compressing video data into a manageable latent space, which is then modeled autoregressively using a Transformer-based architecture. 

This method provides a balance between computational efficiency and the ability to generate detailed and diverse video content, improving upon the limitations of previous generative models for video.

As a pioneering work, the paper is not published anywhere yet, interestingly. Maybe due to lack of SOTA results. The design is simplistic, with single scale of discrete latents and transformers for the autoregressive priors, a design choice also adopted in DALL-E.

The idea is further adopted by [GAIA-1](gaia_1.md) and [VideoPoet](video_poet.md). VideoGPT is not yet action-conditioned yet, so it is NOT a world model (yet).

#### Key ideas
- Model architecture: two-stage process 
	- VQ-VAE for learning compressed latent representations
	- A Transformer model for generating video sequences from these latents.
- Eval: Frechet Video Distance (FVD) and Inception Score (IS)

#### Technical details
- Important training details include using EMA updates for VQ-VAE for faster convergence and employing axial attention within the Transformer to improve sample quality.
- Highlights the challenge of video generation compared to image or text due to the additional temporal complexity and higher computational requirements.

#### Notes
- NA
