# [3D-GCK: Single-Shot 3D Detection of Vehicles from Monocular RGB Images via Geometrically Constrained Keypoints in Real-Time](https://arxiv.org/abs/2006.13084)

_August 2020_

tl;dr: Annotate and predict an 8DoF polyline for 3D perception.

#### Overall impression
The paper proposed a way to annotate and regress a 3D bbox, in the form of a 8 DoF polyline. This is very similar but different from [Bounding Shapes](bounding_shapes.md).

This is one of the series of papers from Daimler.

- [MergeBox](mb_net.md)
- [Bounding Shapes](bounding_shapes.md)
- [3D Geometrically constraint keypoints](3d_gck.md)


#### Key ideas
- Bounding shape is one 4-point and 8DoF polyline. 
![](https://cdn-images-1.medium.com/max/1600/1*6wnwtLdXQ9WcrxTioK4DCw.png)
- Distance is calculated with IPM.

#### Technical details
- Summary of technical details

#### Notes
- Series production cars: mass production cars

