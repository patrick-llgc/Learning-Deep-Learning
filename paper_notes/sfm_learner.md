# [SfMLearner: Unsupervised Learning of Depth and Ego-Motion from Video](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/cvpr17_sfm_final.pdf)

_June 2019_

tl;dr: Unsupervised learning framework to learn monocular depth and camera motion (6-DoF transformation) simultaneously. Use view synthesis and **consistency** as the supervision (similar to stereo depth estimation).

#### Overall impression
One way to do unsupervised learning is through stereo pairs, and the other way to do it is from monocular video frames. This paper ensures consistency with very little assumption (intrinsic matrix is assumed).

The idea is similar to the cycle consistency of cycleGAN as well.

#### Key ideas
- Assumption:
	- No occlusion/dis-occlusion between neighboring frame
	- Environment is static
	- Lambertian (pixel value is similar regardless of angle)
- The scene is assumed to be static. To tackle with moving objects, explainability map is used. 
- Use a single frame to estimate depth, with DispNet architecture (similar to Unet).
- Use all but one frames to estimate the pose and explainability mask, with the same U-Net architecture. 
- Loss: 
	- View synthesis loss is the difference between the synthesized view and the target view, which incorporates the Depth and Pose in the transformation. This loss is modulated by E, explainability mask.
	- Smoothness loss $|\nabla D|^2$
	- Regularization of map E, with CE(E, 1) to encourage every pixel has confidence 1. 
- **Note that the depth estimation is only accurate up to a scalar.** The absolute depth is not determinable from stereo either. We need some sparse measurement to get this fixed to absolute depth.

#### Technical details
- CE is used to regularize explainability mask. This idea should be able to generalize to other processes as well.
- All the depth are positive numbers, so the paper actually used $\frac{1}{10\sigma+0.1}$ to make sure the range is within [1/10.1, 1/0.1]. But why did they use the 1/x for? Using ${10\sigma+0.1}$ can achieve similar effect. 

#### Notes
- New ideas: use the explainability map for object detection. For anchor based method, if we predict the confidence/explainability map, we perhaps could try to use all the anchors for prediction. Another way to use this idea is to build on top of [FCOS](fcos.md), and if we predict this map we do not need the kludgy centerness score. 
- We see again the use of CE as loss for numbers between 0 and 1. Of course we could use the more general focal loss.
- Q: why the 1/x in the distance transformation?
- Q: in training all source views (pre- or post-target) frames are used. But in inference we only have the previous frames?

#### Related works
- LEGO (SOTA for unsupervised on static scene)
	- Learns surface normals and edges
	- Interesting losses, but not implemented?
	- Q: IMU is measurable, why use DL?
	- sample in space, not in time (depending on speed)
	- Manually removed static scnes
	- Project relative static object to infinity
- Struct2Depth
	- Offline mask rcnn to remove the dynamic object
	- Estimate pose and depth from masked image
	- Tell ego pose from static objects first
	- then infer other dynamic objects
- In the wild (SOTA for dynamic scenes)