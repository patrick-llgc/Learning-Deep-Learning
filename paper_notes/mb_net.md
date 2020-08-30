# [MB-Net: MergeBoxes for Real-Time 3D Vehicles Detection](https://ieeexplore.ieee.org/document/8500395)

_August 2020_

tl;dr: Use 5DoF 2d bbox to infer 3d bbox.

#### Overall impression
The paper proposed a way to annotate and regress a 3D bbox, in the form of a 5 DoF bbox (MergeBox). 

This is one of the series of papers from Daimler.

- [MergeBox](mb_net.md)
- [Bounding Shapes](bounding_shapes.md)
- [3D Geometrically constraint keypoints](3d_gck.md)

#### Key ideas
- A 5 DoF bbox to represent 3d bbox.
![](https://cdn-images-1.medium.com/max/1600/0*zsn-mMUfeeUejb1t.png)
- 3D car size templates have to be assumed to lift the mergebox representation to 3D. 

#### Technical details
- The authors noted that even one single template can achieve good performance for AOS (average orientation score). 

#### Notes
- The fancy name for (cos(theta), sin(theta)) is called Biternion. The gaussian on unit circle is called von Mises distribution.
- 3D annotation generally has two approaches: using lidar or 3D CAD model.
- This is similar to what nvidia does by marking the visible edges of the car.

