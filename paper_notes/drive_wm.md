# [Drive-WM: Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving](https://arxiv.org/abs/2311.17918)

_February 2024_

tl;dr: First consistent, controllable, multiview videos generation for autonomous driving.

#### Overall impression
The main contribution of the paper is **multiview** consistent with video generation, and the application of this world model to planning, through **a tree search**, and **OOD planning recovery**.

Drive-WM generates future videos, conditioned on past videos, text, actions, and vectorized perception results, x_t+1 ~ f(x_t, a_t). It does NOT predicts actions. In this way, it is very similar to [GAIA-1](gaia_1.md), but extends GAIA-1 by multicam video generation. It is also conditioned on vectorized perception output, like [DriveDreamer](drive_dreamer.md).

[Def] In a broad sense, a world model is a model that learns a general representation of the world and predicts future world state resultsing from a seq of actions. In the sense of autonomous driving and robotics, a world model is a video prediciton model conditioned on past video and actions. In this sense, Sora generates videos conditioned on input of text and video. But qualitative actions can be expressed as text, so Sora can also quality as a world model. The usage of world model is two fold: to act as a neural simulator (for closed loop training), and to act as a strong feature extractor for policy tuning finetraining.

Video prediction can be regarded as a special form of video generation, conditioned on past observation. If the video prediction can be controled by actions (or in the qualitative form of text), then the video prediction model is a world model.

For a (world) model that does not predict actions, it may act as a neural simulator, but it may not generate enough representation to be finetuned for policy prediction.

It seems that the world model heavily depend on external crutches such as view factorization, and BEV layout. It does NOT learn geometric consistencies through large model traning like [GAIA-1](gaia_1.md) or Sora.

#### Key ideas
- Multiview Video Diffusion Model
	- Image diffusion model, trained first. Initialized from Stable Diffusion ckpt.
	- Temporal encoding layers (as in VideoLDM), and multiview encoding layers. These two are trained later with image diffusion model frozen.
	- Principle, to ensure consistency across one dimention (temeporal or cross camera) there must be info exchange along that dim.
- **Factorization** of joint multiview modeling
	- Divde all frames into reference (anchor) views and stiched (interpolated) views. Views belonging to the same type do not overlap with each other and can be generated independently. --> Sort of, actually some cars will span across more than 2 camras.
	- The factorization is a good engineering trick to ensure cross camera consistency and only applies to the multicam config on autonomous driving cars. -->  It is NOT general enough to guarantee geometric consistency in a wider sense.
	- The factorization significantly boost the multiview consistency in terms of point matching.
- Unifined conditional generation
	- initial context frames, text, ego action, 3D bboxes, BEV maps are used to condition/control the generation of multiview videos. All of the info are convereted to d-dim features and concatenated.
	- BEV Layout is first projected to image space and each class rendered in diff colors. --> BEV Layout is very important to the model's high performance. This means that the video does not necessarily learns the physics rules of the world, such as cars cannot hit the curb, etc.
	- The generation of future frame videos are not conditioned on previously generated videos. --> **The video generation is NOT autoregressive**, due to training limitations. Yet the generated frames are implicitly interdependent through the temporal self-attention in the main network.
- WM for planning via Tree-based rollout with actions. 
	- For each sampled action, future frame is generated.
	- Based on the generated frame, perception is performed and the results are evaluated in terms of map reward (away from the curb and stick to centerline) and object reward (away from other objects).
	- Action with max reward is selected, and the planning tree forward to the next step and plans subsequent trajectory.
	- There are 3 command used to explore the planning tree: turn left, turn right and go straight. --> This is a bit too coarse.
- OOD recovery by finetuning planner with generated OOD videos with supervision by the trajectory that the ego drives back to the lane. 
- Data curation
	- Training dataset is rebalanced by re-sampling rare ego actions. The speed and steering angle are divided into multiple bins, and the data are sampled. 
- Eval
	- multicam consistency: keypoint matching score. 
	- Video quality: FID (Fréchet Inception Distance) and FVD.

#### Technical details
- It shows an example of how to drive on a road with puddles, very similar to the [tesla FSD V12 demo](https://x.com/AIDRIVR/status/1760841783708418094).

#### Notes
- The paper includes a nice summary of the Dreamer series paper on P3.
- The code initiates from that of [VideoLDM](video_ldm.md) (Align Your Latents). 
- The cross view consistency is very nice (showcaing the effectiveness of factorized multiview modeling), but the temporal consistency is not that great, with the appearance of vechiles change througout the video. This may be related to the fact that the future frame geenration is only conditioned on the first frame but not genrated frame. 
- Q: I wonder how much the controllablity from action is from the BEV vectorized results. --> The BEV layout was given as a static resutls and will not change with diff action. So indeed the video generation is conditioned on the action. Yet the action is very hard to learn, and can only be learned when video difusion model is convered. 