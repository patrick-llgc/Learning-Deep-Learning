# [Learning to Look around Objects for Top-View Representations of Outdoor Scenes](https://arxiv.org/abs/1803.10870)

_September 2020_

tl;dr: Hallucinate occluded areas in BEV, and use simulation and map data to help.

#### Overall impression
This is a seminal paper in the field of BEV semantic segmentation, but it does not seem to have received much attention.

This paper goes down a different path compared to [Cam2BEV](cam2bev.md). In [Learning to look around objects](learning_to_look_around_objects.md), the network is explicitly supervised to hallucinate, whereas [Cam2BEV](cam2bev.md) eliminates the occlude regions in order to make the problem better posed.

The paper predicts semantic segmentation and depth in order to lift perspective images to BEV. In this sense it is very similar to [Lift, Splat and Shoot](lift_splat_shoot.md). It also uses a BEV refinement module to refine the intermediate imperfect BEV map. This is very similar to [BEV-seg](bev_seg.md). --> "Depth and semantics are all you need?"

Human supervision in BEV space is hard to procure. Thus this paper used adversarial loss to make sure the BEV layout looks like a real one. It is very similar in idea to [MonoLayout](monolayout.md). but a bit different from [BEV-seg](bev_seg.md).

#### Key ideas
- **View transformation**: pixel-wise depth prediction
- Learn to hallucinate (predict occluded portions).
	- For dynamic objects, filter out loss, as GT loss is hard to get.
	- Randomly masking out blocks of images and ask model to hallucinate. Use the loss as supervision signal.
	- This hallucination of occluded region happens in depth and semantic space, rather than in RGB space.
	- The results from this first stage is an occlusion-reasoned, imperfect BEV.
- Learn priors and rules about typical road layout
	- Adversarial loss to tell if the predicted BEV map makes sense (with the help from synthetic data). --> referred to as "Refinement with a knowledge corpus"
	- Use OpenStreeMap to align to geo-tagged image, and then provide supervision.
	- Loss
		- Reconstruction loss: imperfect BEV and finetuned BEV should match
		- Adversarial loss: finetuned BEV should be reasonable (similar to simulated road layout)
		- Reconstruction loss OSM: after warping, the OSM layout should match that of the finetuned BEV
- Only a single RGB image is required at inference time. Adversarial and map data are only used during training.

#### Technical details
- Indoor scene understanding can rely on strong assumption like a manhattan world layout.
- The foreground dynamic objects are not only masked out, but also the masks are explicitly encoded and send to neural network as input.
- If adversarial loss weight is too large, then the network will predict a simulation prior. If too small, it will be exactly the same as the imperfect BEV.

#### Notes
- I feel neural network can indeed do a good job in hallucination, and the occlusion preprocessing is not necessary for training. The occlusion reasoning can be useful for evaluation.

