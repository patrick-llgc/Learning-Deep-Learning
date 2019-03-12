# [Frustum PointNets for 3D Object Detection from RGB-D Data](https://arxiv.org/pdf/1711.08488.pdf) (F-PointNet)
_Mar 2019_

tl;dr: Combines mature 2D object detection and advanced 3D deep learning for 3D object detection.

#### Overall impression
This paper builds on top of the seminal [pointnet](pointnet.md) for segmentation, and combines with 2D object detection to build a 3D  object detection pipeline.

#### Key ideas
- Data conversion: depth image to point cloud. Note that point cloud does not 
- Extrude the 2D bbox from 2D object detection into 3D space. 

#### Technical details
- Summary of technical details

#### Notes
- Lidar point cloud data captures 3 degrees of dimension: Azimuth, height/elevation/altitude, and distance/radius (according to the spherical coordinate system).

