# [MonoLayout: Amodal scene layout from a single image](https://arxiv.org/abs/2002.08394)

_June 2020_

tl;dr: Predict BEV semantic maps from monocular images.

#### Overall impression
This is very similar to [PyrOccNet](pyroccnet.md). 

[monolayout](monolayout.md) uses self-generated ground truth by aggregating results throughout video (so-called *temporal sensor fusion*). HD Map GT is only used for evaluation.

The authors also listed tricks that did not work. This I think should be the recommended standard practice in future!

#### Key ideas
- Detached dynamic layout and static layout. 
	- Dynamic layout: this is more related to mono 3D MOD. 
		- Instance label
	- Static layout is more related to [what Tesla is doing](../talk_notes/andrej.md). 
	- Network predicts static or dynamic layout whether it is covered by the camera or not. This is quite different from the method used in [PyrOccNet](pyroccnet.md) where occluded points are masked. 
- Architecture
	- One encoder, two decoder (dynamic + static)
		- The learned representation must implicitly disentangle the static parts and dynamic objects. 
	- patch based discriminators
		- Plausible road geometries extracted from **unpaired** database of **openstreetmap**. 
- **Generating training data** via temporal sensor fusion
	- Use monodepth2 or lidar to lift RGB to point cloud. 
	- With odometry info, aggregate and register the scene observation over time, to generate a more dense, noise free point cloud. 
	- When using monodepth2, discard anything 5 m away from the ego car as they could be noisy. 
	- Aggregate 40-50 frames. 
	- Use GT or predicted semantic labels and aggregate into occupancy grid by majority voting.
- Compare with pseudo-lidar, monolayout can achieve equal or better results but much faster. 
- This work is easily extended to be converted to a behavior predictor. 

#### Technical details
- 40 x 40 m, 128 x 128 grid. 
- Realtime, 30Hz on GTX 1080 Ti.
- Argoverse contains high-res semantic occupancy grid in BEV. 
- Things the authors tried but did not work
	- Using a single decoder to decode both dynamic and static layout.
- Drawbacks: shadows will make the network into predicting protrusions along the shadow direction. 

#### Notes
- [Talk at WACV 2020](https://www.youtube.com/watch?v=HcroGyo6yRQ)
- [Code on github](https://github.com/hbutsuak95/monolayout)
