# [WorldDreamer: Towards General World Models for Video Generation via Predicting Masked Tokens](https://arxiv.org/abs/2401.09985)

_March 2024_

tl;dr: Multimodal world model via masked token prediction.

#### Overall impression
The model takes in a variety of modalities such as image/video, text, actions, and generate videos conditioned on these multimodal prompts.

World models hold great promise for learning motion and physics in the genral world, essential for coherent and reasonable video generation. It draws strong aspiration from [VideoPoet](video_poet.md) and adds action condition on top of it, making VideoPoet a world model.

[WorldDreamer](world_dreamer.md) seems to be the extension of [DriveDreamer](drive_dreamer.md). Yet disappointingly [WorldDreamer](world_dreamer.md) seems unfinished and rushed to release on Arxiv, without much comparison with contemporary work. The paper is also heavily inspired by MaskGIT, especially the masked token prediction and parallel decoding.

#### Key ideas
- Architecture
	- Encoder
		- Vision: VQ-GAN, vocab = 8192
		- Text: pretrained T5, similar to [GAIA-1](gaia_1.md).
		- Action: MLP
		- Text and action embedding can be missing.
	- Masked prediciton
	- Decoder: Parallel decoding
- Training with masks.
	- Dataset: triplet (visual, text, action), but also supports data with missing modalities.
- Inference: parallel decoding
	- DIffusion: requires ~30 steps to reduce noise
	- Autoregressive: needs ~200 steps to iteratively predict next token
	- Parallel decoding: video generation in ~10 steps.


#### Technical details
- The key assumption underlying the effectiveness of the parallel decoding is a Markovian
property that many tokens are conditionally independent given other tokens. (From [MaskGIT](https://masked-generative-image-transformer.github.io/) and Muse)
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) to detect scene switching
- The idea of using masked language model for video prediction is first proposed in MaskGIT, then extended by Muse to text-to-image generation. During training, MaskGIT is trained on a similar proxy task to the mask prediction in BERT. At inference time, MaskGIT adopts a novel non-autoregressive decoding method to synthesize an image in constant number of steps.

#### Notes
- Questions and notes on how to improve/revise the current work

