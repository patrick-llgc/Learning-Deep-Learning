# [CasGeo: 3D Bounding Box Estimation for Autonomous Vehicles by Cascaded Geometric Constraints and Depurated 2D Detections Using 3D Results](https://arxiv.org/abs/1909.01867)

_October 2019_

tl;dr: Extends Deep3DBox by regressing the 3d bbox center on bottom edge and viewpoint classification. 

#### Overall impression
CasGeo predicts more geometric keypoints or properties from the image patch. It regresses projection of the center of 3D bbox bottom face (CBF) and viewpoint classification. 

It uses similar constraint such as GS3D to regress the 3D height of the bbox in 2D image. But instead of using the statistical mean of 93% of 2d bbox height, CasGeo regresses this distance from image patch. Then there are two steps: first approximate the location such as GS3D does, then feed into the Gauss-Newman method to solve the over-constraint problem proposed by deep3dbox.

The paper also removes false positive in 2D detection based on inferred 3D information.

#### Key ideas
- Center of 3d bbox bottom face (CBF) is used to estimate a good initial guess of 3d position
- Viewpoint regression helps picking which point to use to correspond to the four edges of the 2d bbox. There are 16 viewpoints corresponds to 4 top selection x 4 bottom selection x 4 rot90 = 64 configurations.

#### Technical details
- Optimization using Gauss-Newman method to solve the over-determined problem --> this is not necessary at all as the problem Ax = b is convex and has a global minimum. 
- Using pseudo inversion to solve the problem seems sufficient. 

#### Notes
- Can we extend the method to use the backplane edges into the equations? The over-constraint problem only solves the noise in 2d bbox detection, but there are noise in yaw and vehicle size estimation as well. --> this seems to be a sensor fusion problem.
