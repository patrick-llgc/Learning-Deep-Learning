# [BS3D: Beyond Bounding Boxes: Using Bounding Shapes for Real-Time 3D Vehicle Detection from Monocular RGB Images](https://ieeexplore.ieee.org/abstract/document/8814036/)

_August 2020_

tl;dr: Annotate and predict a 6DoF bounding shape for 3D perception.

#### Overall impression
The paper proposed a way to annotate and regress a 3D bbox, in the form of a 8 DoF polyline. 

This is one of the series of papers from Daimler.

- [MergeBox](mb_net.md)
- [Bounding Shapes](bounding_shapes.md)
- [3D Geometrically constraint keypoints](3d_gck.md)

#### Key ideas
- Bounding shape is one 4-point and 8DoF polyline. 
![](https://cdn-images-1.medium.com/max/1600/0*D4Rm2BD-MbN1dx9W.png)


#### Technical details
- The paper normalizes dimension and 3D location to that y = 0. When real depth is recovered (via lidar, radar or stereo), the monocular perception 
- An object is of [Manhattan properties](https://openaccess.thecvf.com/content_cvpr_2017/papers/Gao_Exploiting_Symmetry_andor_CVPR_2017_paper.pdf) if 3 orthogonal axes can be inferred, such as cars, buses, motorbikes, trains, etc.

#### Notes
- 57% of all American drivers do not use turn signals when changing the lane. ([source](https://www.insurancejournal.com/news/national/2006/03/15/66496.htm))

