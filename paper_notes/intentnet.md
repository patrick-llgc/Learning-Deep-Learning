# [IntentNet: Learning to Predict Intention from Raw Sensor Data](http://www.cs.toronto.edu/~wenjie/papers/intentnet_corl18.pdf)

_April 2020_

tl;dr: A boosted version of [Fast and Furious](faf.md) that uses map information.

#### Overall impression
[IntentNet](intentnet.md) is heavily inspired by [Fast and Furious](faf.md) (also by Uber ATG). Both combines perception, tracking and prediction by generating bbox with waypoints. In comparison, IntentNet extends the horizon from 1s to 3s, predicts discrete high level behaviors, and uses map information. Note that both [IntentNet](intentnet.md) and [Fast and Furious](faf.md) do not perform motion planning.

The concatenation of images may hurt the model's real time performance as inferenece time is not reported. FaF uses LSTM 

#### Key ideas
- Map info rendered as 17-channel binary BEV map. 
- Architecture: Late fusion of map and lidar (first 3 blocks split, with the last block concatenated)
	- Concatenate past 10 frames along channel dimension --> cannot recycle computation. Why don't use LSTM for better real-time performance?
	- Predict discrete intention (8 classes)
	- Predicts current location/detection
	- Predicts future waypoint. 
- Loss (FaF-like, but with time discount)
	- YOLO-like, anchor-based object classification (binary focal loss). . 
	- Trajectory regression of waypoint positions, orientation, and w and h (constant over time) (smooth L2 loss)
	- Loss is discounted by 0.97 each timestep into the future

#### Technical details
- 144 x 80 meters (0.2 m per pixel), 720 x 400 pixels. 

#### Notes
- The paper argues that using intermediate perception and tracking results for intent prediction restricts the information and may have suboptimal performance. --> This may be true in theory but in practice how much effect the image information has to intent prediction is questionable. The ultimate question is, can human being figure out the intent just by the perception results?

