# [BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation](https://arxiv.org/abs/2205.13542)

_June 2022_

tl;dr: Early camera-lidar fusion in BEV space.

#### Overall impression
BEVFusion breaks the long-lasting common practice that point-level fusion is the golden choice for multi-sensor perception systems.

BEVFusion's main contribution seems to be the efficient implementation of the Voxel Pooling operation. The speed bottleneck in voxelpooling is also noted and solved in [BEVDepth](bevdepth.md).

BEVFusion is single image based, no temporal module.

#### Key ideas
- Advantages of different sensors: Camera capture rich **semantic information**, lidar provide accurate **spatial information**, and radar offers **instant velocity estimation**.
- Projecting lidar to camera (RGBD data format) is a **geometrically lossy** process, as two neighbors on the depth map can be faraway from each other in 3D space. Projecting camera to lidar is a **semantically lossy** process, as point level fusion throws away the semantic density of camera features.
- 3D Object detection is a geometric-oriented task, map segmentation is semantic-oriented.
- Network acceleration. BEV voxel pooling is surprisingly slow.
	- Grid association: The mapping between each camera pixel and BEV voxel is fixed (as the distance bin is also fixed), and therefore can be precomputed. [2X Speedup]
	- Internal reduction: all points within the same BEV are aggregated with some symmetric function (mean, max, sum). [LSS](lift_splat_shoot.md) used the cumsum trick based integral image, but this way is not efficient. To accelerate this, BEVFusion directly assigns a GPU thread to each grid and process that cell. [20X Speedup]
- Fully convolutional fusion
	- Camera BEV features may be misaligned with lidar BEV features. The convolutional layers in the BEV encoder can compensate for such local misalignments.

#### Technical details
- **Jointly training** different tasks together (detection of dynamic objects and semantic segmentation of static objects) has a negative impact on the performance of each individual task ("negative transfer"). --> This is also observed in [M2BEV](m2bev.md) and [PETRv2](petrv2.md).
- The night time performance of camera only performance (BEVFusion version) is worse than lidar-only performance (center point). --> **This is a major risk of BEVNet in production.**

#### Notes
- Questions and notes on how to improve/revise the current work
