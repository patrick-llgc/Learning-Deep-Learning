# [HDNET/PIXOR++: Exploiting HD Maps for 3D Object Detection](http://proceedings.mlr.press/v87/yang18b/yang18b.pdf)

_November 2020_

tl;dr: Exploiting HD map priors to guide 3D lidar object detection.

#### Overall impression
The paper is very well written with clear logic. 

Maps contain geographic, geometric and semantic priors that re useful for many tasks. While HD maps are widely used by motion planning, they are vastly ignored by perception systems. 

The PIXOR++ baseline can already beat previous lidar-only method such as [PIXOR](pixor.md), [MV3D](mv3d.md) and sensor fusion methods such as [ContFuse](contfuse.md).

#### Key ideas
- PIXOR++: improvement over [PIXOR](pixor.md).
	- Input representation: two additional channels to capture points outside height range.
	- ground height: replace absolute distance z with distance relative to the ground. 
	- road mask: extract road layout information from HD maps and rasterize it to the same resolution as discretized Lidar representation.
- Online mapping:
	- One U-Net for ground height estimation
	- One U-Net for road mask segmentation
- Training with data dropout on the semantic prior to increase the robustness to the online availability of HD maps. --> This is similar to the camera dropout in [Lift Splat Shoot](lift_splat_shoot.md).
	- When map is not available, the performance is almost as good as baseline. 


#### Technical details
- Modification of regression target as compared to [PIXOR](pixor.md)
	- Regress $\sin(2\theta)$ and $\cos(2\theta)$ with a period of $\pi$ as 180 deg rotation is the same for IoU evaluation.
	- Regress dx, dy, logw and logl.
	- filtering out loss at empty locations during training leads to better perf.
- Pixel wise ground plane estimation is better than ground plane fitting.
	- Ground plane fitting helps with detection only in the hard case, but actually hurts easy and moderate cases.
- Benefit of offline map increases with distance. The benefit of online map first increases and decreases with distance, due to the difficulty for accurate map generation at far distance online. 

#### Notes
- Using offline map to improve perception may be a secondary or third degree order problem in a perception stack. Using online mapping capabilities is more practical and scalable. 
- The ground height and road mask segmentation GT can be obtained with a self-supervised way, as in [MonoLayout](monolayout.md).
- A typical autonomous system is composed of the following functional modules: perception, prediction, planning and control (PNC). 
	- Perception is concerned with detecting the objects of interest (e.g. vehicles) in the scene and track them over time. 
	- The prediction module estimates the intentions and trajectories of all actors into the future. 
	- Motion planning is responsible for producing a trajectory that is safe, while control outputs the commands necessary for the self-driving vehicle to execute such trajectory

