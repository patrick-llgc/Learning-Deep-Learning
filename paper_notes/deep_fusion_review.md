# [Deep Multi-modal Object Detection and Semantic Segmentation for Autonomous Driving: Datasets, Methods, and Challenges](https://arxiv.org/pdf/1902.07830.pdf)

_July 2019_

tl;dr: Very good review of sensor fusion in 3D object detection.

#### Overall impression
The summary of data fusion schemes is highly informative. After all, it seems that the pioneering work of [MV3D](mv3d.md) and [AVOD](avod.md) are still highly relevant.

Deep learning on Radar data has not been widely explored yet and have potential to yield better object detection algorithms.

#### Key ideas
- For 3D object detection, PointPillars has the best balanced algorithm based on lidar-only, and AVOD is the best balanced algorithm based on lidar/camera fusion.
- Three ways to process lidar point cloud
	- voxelization
	- pointnet
	- projection to BEV or CPM (camera plane map) or spehrical map (azimuth and zenith angles)
- Project cameras to BEV space can help with occlusion and scale variations.

#### Technical details
- FMCW lidars can provide speed info. Lidars are less affected by fog, rain than visual cameras, but still is not robust enough than radar.
- Classifying objects in radar is highly challenging due to low resolution.
- MVLD dataset has intention and interaction labeled. 
- For sensor fusion, **well calibrated sensors with accurate spatial and temporal alignment is the prerequisite for accurate multi-model perception**.

#### Notes
- New ideas:
	- Learn multi-modal fusion without accurate calibration
	- Monocular depth estimation with BEV projection


