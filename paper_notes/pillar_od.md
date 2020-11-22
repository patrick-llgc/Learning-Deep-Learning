# [Pillar-based Object Detection for Autonomous Driving](https://arxiv.org/abs/2007.10323)

_November 2020_

tl;dr: Improved [multi-view fusion](mvf.md).

#### Overall impression
Three key improvements based on [MVF](mvf.md). The ablation studies in this paper is super clean and persuasive. 

#### Key ideas
- Multiview architecture
	- Voxelize points in BEV or spherical view or cylindrical view to pillars.
	- Extract pillar features.
	- Project pillar features to points with nearest neighbor or bilinear interpolation and concat to point features. 
	- Transform point features to BEV
	- Detection backbone + head
![](https://miro.medium.com/max/942/0*dzktcm4hQwKh1oup.png)
- **Anchor-based Pillar-based prediction**: like [CenterPoint](centerpoint.md) and [Pixor](pixor.md).
	- Both [PointPillars](point_pillars.md) and [MVF](mvf.md) uses anchor-based prediction.
	- Anchor free avoids complicated anchor matching strategy.
	- Ablation studies show that anchor-based < point-based << pillar-based.
- **Cylindrical view**: height z, azimuth angle, radial distance. The radial distance is treated as channels. 
	- Cylindrical view is better than spherical view as the vehicle size for distant cars are not distorted. Distant cars appears smaller in spherical view but the same in cylindrical view. --> [LaserNet](lasernet.md) uses a range view (RV) which is very similar to spherical view. The original [MVF](mvf.md) is also a spherical view. 
- **Bilinear upsampling** when transferring pillar features to point. 
	- This avoids the spatial inconsistency and dependency of quantization into diff bins. 
	- Bilinear interpolation is better than nearest neighbor. This observation is consistent with the comparison between RoIAlign with RoIPooling.

#### Technical details
- Positive anchors in lidar BEV is very sparse (< 0.1%).

#### Notes
- Questions and notes on how to improve/revise the current work  

