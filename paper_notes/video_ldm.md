# [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818)

_March 2024_

tl;dr: First video generation pipeline based on latent space.

#### Overall impression
Two main advantages of video LDM is the computationally efficiency, and ability to leverage pretrained image diffusion models. Video LDM leverages pretrained image DMs and them into vido generators by inserting temporal layers to enforece temporally coherent reconstruction. It is the first video diffusion model in latent space rather than in pixel space.

> Diffusion models offer a robust and **scable** training objectave and are typically less parameter intensive than their transformer based counterparts. 

> Latent diffusion models works in a compressed lower dim latent space and thus makes the task of high-res video generation more tractable.

It is also cited by Sora as one comparison baseline. Video LDM is widely used in research projects due to its simplicity and compute efficiency.

The temporal consistency of the long drive video is still not very good, without fixed appearances for a given object. Similar to that in [Drive into the Future](drive_wm.md).

Video generation displays multimodality, but not controllability (it is conditioned on simple weather conditions, and crowdedness, and optionally bbox). In this sense it is NOT a world model.


#### Key ideas
- Adaptation Procedure from Image LDM to Video LDM
	- Initialize the models from image LDMs --> It can leverage different diffusion models, and can benefit from the advances in this field.
	- Insert temporal layers into the LDMs' denoising neural networks to temporally model encoded video frame sequences. The temporal layers are **interleaved** with the existing spatial layers. The temporal layers are based on temporal attention as well as 3D convolutions. 
	- Generate key frames
	- Latent frame interpolation temporally for higher frame rates. 
	- Decode into pixel space with finetuned decoder for video generation
	- Apply video superresolution upsampler. 
- Video finetuning of Autoencoder decoder and Superresolutionn models
	- SR video frames independently would results in poor temporal resolution, so SR module needs to be video aware.
	- AE is used to to map image into latent space
	- Encoder is frozen from image AE
	- Decoder is finetuned for video. This step is critical to achieve better FVD scores.
- The learned temporal layers **transfer or generalize** to different model checkpoints. 

#### Technical details
- Video generation models "help democratiz artistic expression".
- End-to-end video LDM training suffers from performance loss compared with init with pretrained image LDM models.
- In generating realistic driving scenario, video LDM also trained a separate bbox conditioned image LDM. --> This perhaps heavily inspired [DriveDreamer](drive_dreamer.md).
- FVD is not reliable. FVD is sensitive to the realism of individual frames and motion over short segments, but that it does not capture **long-term realism**. For example, FVD is essentially blind to unrealistic repetition of content over time. We found FVD to be most useful in ablations, i.e., when comparing slightly different variants of the same architecture.(See more details in Section 5.3 of Long-Video-GAN)
- Personalized text-to-video with DreamBooth. The use case is to preserve object identify. Specifically, given several images of one particular object, LDM can leverage the pretrained DreamBooth image ckpt and generate videos for the given object.

#### Notes
- Q: The notion of latent space is widely used. The latent space in latent diffusion model, and the latent space in world model such as [Genie](genie.md), and the latent representation needed for action prediction. Are they the same? How are they related? 
- The model also exhibits multi-stage traning, with image-specific training and video finetuning, similar to the training schedule of mass production BEV temporal models.
