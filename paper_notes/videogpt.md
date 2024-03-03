# [VideoGPT: Video Generation using VQ-VAE and Transformers](https://arxiv.org/abs/2104.10157)

_March 2024_

tl;dr: Uses VQ-VAE and Transformers to efficiently and  autoregressively generate high-fidelity videos from textual descriptions or other conditional inputs.

#### Overall impression
The paper's main contribution lies in its novel use of VQ-VAE for compressing video data into a manageable latent space, which is then modeled autoregressively using a Transformer-based architecture. 

> Natural images and videos contain a lot of spatial and temporal redundancies and hence the reason we use image compression tools such as JPEG (Wallace, 1992) and video codecs such as MPEG (Le Gall, 1991) everyday. --> This is also echoed in [latent diffusion models](ldm.md). Most bits of a digital image correspond to imperceptible details. We can perform perceptual compression up to a great ratio without compressing semantics. 

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
