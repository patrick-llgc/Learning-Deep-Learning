# [How Do Neural Networks See Depth in Single Images?](http://openaccess.thecvf.com/content_ICCV_2019/papers/van_Dijk_How_Do_Neural_Networks_See_Depth_in_Single_Images_ICCV_2019_paper.pdf)

_December 2019_

tl;dr: Probes the monodepth estimator as blackboxes and see how the different estimators reacts to changes of different geometric cues. 

#### Overall impression
The paper performs the missing "ablation study" for the monocular depth estimators. It discovers that all depth estimators examined uses the vertical position of the object as depth cues. 

**Video-based method such as SLAM or SfM tend to treat depth estimation as pure geometrical problem, ignoring the contents of the image.**

The depth estimation networks learns to find where the object touches ground and fill in the depth in the object contour. It uses a dark, thick edge (shadow below cars) to detect the region where object touches ground.

#### Key ideas
- Pictorial cues for depth estimation:
	- Position in the image: Further away objects are closer to the horizon
	- Occlusion: closer objects occlude further away objects. Only depth order is provided
	- Apparent size of objects: objects that are further away appear smaller.
	- Others: linear perspective, texture density, shading/illumination, focus blur.
- For autonomous driving, two depth cues are applicable: object size, and the bottom position of objects. Object size assumes known size of objects, and vertical image position does not have this assumption.
- The network is using vertical position of the objects (specifically bottom) to estimate depth. It is not looking at the distance to horizon nor object size.
- The depth estimation network network
	- finds the ground contact point of the obstacle
	- finds the outline of the obstacle to fill in the depth region
- **The networks uses a shadow region below the car to detect cars. The bottom edge needs to be both thick and dark for a successful detection.** The network cannot detect foreign obstacles not in training dataset, but once the black thick edge is added to the bottom of the obstacle, they can be recognized. 

#### Technical details
- Depth of a car is estimated via averaging the depth map over a flat region on the front or rear of the car.
- The depth estimation underestimates the pitch and roll change as well.
- The exact color does not strongly affect the depth estimation. Only value does.
- The car is still detected when the interior is removed. This suggests that the network is sensitive to the outline 

#### Notes
- Questions and notes on how to improve/revise the current work  

