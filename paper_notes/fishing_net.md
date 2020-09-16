# [FISHING Net: Future Inference of Semantic Heatmaps In Grids](https://arxiv.org/abs/2006.09917)

_September 2020_

tl;dr: Convert lidar, radar and camera fusion in BEV space.

#### Overall impression
Perception in autonomous driving involves building a representation that captures the **geometry and semantics** of the surrounding scenes.

The BEV (top-down) representation across modalities has multiple benefits:

- interpretable
- easily extensible
- simplifies the task of late fusion

Maybe BEV representation is the ultimate goal for perception. The authors also noted that we need to **add the concept of instance**. This may be necessary to make the output results to be directly consumable by downstream.

[Fishing Net](fishing_net.md) uses BEV grid resolution: 10 cm and 20 cm/pixel. [Lift Splat Shoot](lift_splat_shoot.md) uses 50 cm/pixel. They are both coarser than the typical 4 cm or 5 cm per pixel resolution used by mapping purposes such as [DAGMapper](dagmapper.md).

#### Key ideas
- **View transformation**: MLP (similar to [VPN](vpn.md)).
- Takes in 5 history frames of sensory data (camera, lidar and radar), and predict semantic BEV frames 5 frames into the future. 
- The GT generation is with 3D annotation in lidar, and it mainly focuses on dynamic objects. 
- Priority pooling: VRU (pedestrian, cyclists, motorists) > cars > background
- Lidar input: 8 channels
	- binary occupancy
	- log normalized lidar density
	- max z
	- max z sliced, 0 to 2.5 m every 0.5 m (5 ch)
- Radar input: 6 channels
	- binary occupancy
	- X, Y value of doppler velocity (motion compensated, 2 ch)
	- RCS (radar cross section)
	- SNR
	- Ambiguous doppler interval
- Autolabeling process
	- 3D track bboxes and semantic labels

#### Technical details
- The radar map is quite dense. This is before clustering?
- Lidar/radar uses an U-Net like architecture with skip connection, but camera does not have it as it has one orthogonal feature transform layer. See [VPN](vpn.md).
- NuScenes radar data are too sparse to perform semantic segmentation.

#### Notes
- [talk at CVPR 2020](https://youtu.be/WRH7N_GxgjE?t=1004)

