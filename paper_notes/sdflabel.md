# [SDFLabel: Autolabeling 3D Objects With Differentiable Rendering of SDF Shape Priors](https://arxiv.org/abs/1911.11288)

_September 2020_

tl;dr: Using differentiable rendering for automatic labeling.

#### Overall impression
Use 2D regression to predict NOCS map and a shape vector. The NOCS map can be used with lidar to extract a sparse 3D model, and the shape vector can be used with DeepSDF to decode a 3D model. The compute approximate pose with 3D matching. Then calculate 2D and 3D loss for back-propagation for refinement. 

Previous work such as [3D RCNN](3d_rcnn.md) and [RoI10D](roi10d.md) uses PCA or CAE (conv auto-encoder) to predict the shape of cars. This is not end-to-end differentiable. DeepSDF enables backpropagation onto a smooth shape manifold and is more powerful. 

Autolabel is still not as good as lidar labels, but very close. Closer in performance in BEV rather than 3D, but only BEV should be good enough for autonomous driving. But the 3D drop may not be real as the autolabels are **tight** 3D bbox as compared to KITTI3D lidar labels. 

#### Key ideas
- CSS (coordinate shape space): combination of NOCS and DeepSDF
	- NOCS (normalized object coordinate system). It encodes the pose and shape information, i.e., surface coordinates. With **dense** depth information, 3D pose can be recovered from NOCS.
	- NOCS is a correspondence map
	- NOCS encodes surface iniformation (normal)
	![](https://cdn-images-1.medium.com/max/1600/1*ZbN913AmRDsblCMJhtzC3g.png)
- DeepSDF to embed watertight models into a joint and compact shape space representation.
	- 11 CAD models are embedded into 3D space (3-dim latent code) with DeepSDF. 
- DeepSDF is combined with differentiable rendering so that the surface point is differentiable wrt scale, pose, or latent code.
- Overall workflow
	- predicted z is decoded, and NOCS coordinate is calculated. 
	- Lidar is projected onto the predicted NOCS map.
	- Estimate initial pose and scale by 3D matching.
- Loss
	- 2D Loss: SDF renderer and decoded map to have a rendered NOCS map, and compare with predicted NOCS map.
	- 3D Loss: correspondence with lidar points --> drastic drop in 3D metric when optimizing only in 2D
- Verification: similar to 2D and 3D losses, they can be used for verification. 
		- projective: Mask IoU > 0.7
		- geometric: lidar points within 0.2 m band of surface > 60%

#### Technical details
- **KITTI3D cuboids have a varying amount of spatial padding** and are not tight. Deep learning models trained on these data will learn the padding too.
- Ways to scale up annotation pipeline include better tooling, active learning or a combination thereof.
- Curriculum learning pipeline: to bridge the synthetic-to-real gap. Iteratively add real samples passing sanity check.
	- Rather fast **diffusion** into target domain.
- Parallel domain (acquired by Toyota). But we should be able to use CARLA or vKITTI as well.
- Level of difficulty of a label is measured by pixel size, amount of intersection with other 2D labels, or whether the label is cropped. 
	- Easy: h > 40 pix
	- Moderate: h > 25 pix, and not having IoU > 0.3 with others
- CSS is trained on about 8k patches.
- 6sec to autolabel one instance. 


#### Notes
- [Code on github](https://github.com/TRI-ML/sdflabel)
- The idea seems to be closely related to [DensePose](densepose.md). It densely map canonical 2D coordinates (coorespondence map) to human bodies. But it allow for projective scene analysis up to scale. --> we can fix that with IPM!?
- [NOCS](nocs.md) extended dense coordinates to 3D space.

