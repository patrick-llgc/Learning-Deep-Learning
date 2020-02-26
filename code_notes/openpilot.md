# [openpilot](https://github.com/commaai/openpilot/blob/devel/selfdrive/modeld/models/driving.cc)

_Februrary 2020_

tl;dr: Open sourced autonomous driving software stack from comma.ai.

### Useful links
- [Decoding comma.ai/openpilot: the driving model](https://medium.com/@chengyao.shen/decoding-comma-ai-openpilot-the-driving-model-a1ad3b4a3612)
- [Unhack Openpilot](https://github.com/peter-popov/unhack-openpilot)
- [A tour through OpenPilot](https://medium.com/@comma_ai/a-tour-through-openpilot-a6589a801ed0)


#### Lane Lines
- Each lane line is made of 192 points, spaced at 1 m apart. ([source](https://github.com/commaai/openpilot/issues/872)). `MODEL_PATH_DISTANCE = 192` (from [code](https://github.com/commaai/openpilot/blob/devel/selfdrive/modeld/models/driving.h#L34))
- They seem to do a perspective warping of image first into BEV space with IPM, then predict with 1 m distance apart. 
- Only ego lane, left and right lane lines.


#### Moving Objects
- Uses MDN (mixture density network) to predict cars. 
- [Mixture model usage in MOD prediction](https://github.com/commaai/openpilot/blob/v0.6.3/selfdrive/visiond/models/driving.cc#L117)
> ```
  // Every output distribution from the MDN includes the probabilties
  // of it representing a current lead car, a lead car in 2s
  // or a lead car in 4s ```

- MDN is an old concept from Bishop's book PRML in 1994.
	- [Pytorch tutorial](https://mikedusenberry.com/mixture-density-networks)
	- [Edward tutorial](http://edwardlib.org/tutorials/mixture-density-network)
	- [Tutorial with TF probability](https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca)
	- [David Ha's tutorial with TF](http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/)
- But it seems that the MOD detection is not used for longitudinal control for now. (Only for visualization in UI?) For longitudinal control they use radar signals.


#### Pose
- PoseNet by Alex Kendall to infer pose. However unlike the original poseNet which infers 6DoF within a map, openpilot takes in two images and output the relative motion. 
- According to the official explanation: 

	> modeld also runs the posenet, in models/posenet.dlc. It takes in two frames and outputs the 6-DoF transform between them. It is used for calibration and sanity checking, and is not trained in any particularly magical way.
- Get vanishing point in C code ([`get_vp.c`](https://github.com/commaai/openpilot/blob/v0.6.3/selfdrive/locationd/get_vp.c)). 
	- `get_vp.c` is introduced in v0.5.2 removed in version v0.7. (Maybe deprecated before that?)
	- **TODO** Check out the release of [v0.5.2](https://github.com/commaai/openpilot/blob/v0.5.2/selfdrive/locationd/calibrationd.py) to see how they did online calibration without DL!
- DL-based self-calibration
	- PoseNet Model started appearing in `models` folder starting v0.5.9.
	- PoseNet start publish starting v0.7.1. [code](https://github.com/commaai/openpilot/blob/v0.7.1/selfdrive/modeld/models/driving.cc#L252)
	- Supercombo model is also released in v0.7.1. [twitter](https://twitter.com/comma_ai/status/1219752361800790016?s=20)