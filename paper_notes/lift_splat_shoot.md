# [Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D](https://arxiv.org/abs/2008.05711)

_September 2020_

tl;dr: Predict depth distribution of each pixel for differentiable rendering of a BEV map. 

#### Overall impression
The paper is build on top of quite a few previous work such as [OFT](oft.md), [PyrOccNet](pyroccnet.md), [MonoLayout](monolayout.md) and [pseudo-lidar](pseudo_lidar.md). 

It proposed probabilistic 3D lifting through prediction of depth distribution for a pixel in the RGB image. In a way it proposed a unified lifting method between the one-hot lifting of [pseudo-lidar](pseudo_lidar.md) and the uniform lifting of [OFT](oft.md). This is a trick commonly used in differentiable rendering. --> Actually [Pseudo-Lidar v3](pseudo_lidar_v3.md) also uses this soft rasterizing trick to make depth lifting and projection differentiable. 

The semantic BEV map prediction need to fuse predictions from all cameras into a single **cohesive** representation of the scene. This is **full presentation learning of the entire 360 scene local to the ego vehicle** conditioned exclusively on camera input. The ultimate goal of the BEV map prediction is to learn dense representation for motion planning.

[Fishing Net](fishing_net.md) uses BEV grid resolution: 10 cm and 20 cm/pixel. [Lift Splat Shoot](lift_splat_shoot.md) uses 50 cm/pixel. They are both coarser than the typical 4 cm or 5 cm per pixel resolution used by mapping purposes such as [DAGMapper](dagmapper.md).


#### Key ideas
- **View transformation**: Probabilistic pixel-wise depth prediction
- Lift: probabilistic (and differentiable) 3D lifting.
	- [4, 45] meters, 1 meter bin. Very much like [DORN](dorn.md).
	- Essentially each pixel in (u, v) creates 42 3D points. This is a huge point cloud. 
- Splat: point pillar generation
- Shoot: motion planning. Predict a distribution over K templates.
- This Lift-Splat has 3D structure at initialization. This is better than baseline methods used by [MonoLayout](monolayout.md)

#### Technical details
- "Resolution":
	- Camera images: HxW = 128x352
	- BEV grid: XxY, 200x200 @ 0.5 m/pixel = 100m x 100m
	- Depth resolution: [4, 45] meters @ 1 meter interval.
- Frustum pooling via cumsum trick (integral image)
	- Sum pooling (avg pooling) can be sped up with integral image. So ideally faster than max pooling.
- Robust training
	- Camera dropout during training adds to the robustness --> similar to the input dropout of HD maps of [PIXOR++](pixor++.md).
	- Training with noisy extrinsics leads to more robust network against calibration noise

#### Notes
- Next step is to use video pipeline to boost the depth prediction accuracy. 

