# [MLF: Multi-Level Fusion based 3D Object Detection from Monocular Images](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Multi-Level_Fusion_Based_CVPR_2018_paper.pdf)

_Aug 2019_

tl;dr: Estimate depth map from monocular RGB and use the depth info for 3D object detection.

#### Overall impression
This paper inspired a more influential paper, [pseudo-lidar](pseudo_lidar.md). Especially, Figure 3 basically has the idea of projecting depth map to point cloud, but it was only used for visualization of the detection results. From Fig. 3 it is quite natural to think about object detection with this pseudo-point cloud.

The idea of regressing the 3D location through local features and global features is correct, but the formulation (addition of predictions from the two branches) are doubtful. Why not concat the features?

This paper also separated the highly correlated pair of depth estimation with size estimation. Directly regressing these two together can be ill-posed, as pointed out by [Deep3Dbox](deep3dbox.md).

Overall the idea of the paper is good, but implementation is not optimal. 


#### Key ideas
- Multi-level fusion
	- The paper directly concats the depth estimation with RGB to form RGB + (depth z, height, distance) 6-ch image and perform object detection on it (input fusion).
	- The second branch of a point cloud (XYZ map) gives global features for location regression, and contributes to feature fusion.
	- The third fusion step is via adding the predictions from max ROIpooled local features + mean ROIpooled XYZ (global) features.
	- The paper also investigated a baseline without the first two fusion steps. --> This is similar to feed in network the bbox locations. 
- The paper also investigated using stereo as input, and the precision is much better than mono. That means the current bottleneck of the model is still the accuracy of depth estimation. This is the same conclusion as the pseudo-lidar paper.
- The way the generated point cloud is used is suboptimal. This inspired later works such as [pseudo-lidar](pseudo-lidar.md). 

#### Technical details
- **Vehicle sizes are not regressed directly**, but rather estimated via offset with mean sizes with local ROIPooled features --> However Eq. 5 can be simplified and you can see it is equivalent to regressing the absolute dimension directly! 

#### Notes
- Pseudo-lidar points are only used to provide x, y, z info, which may be replaced just by feed into bbox center location (u, v) and bbox size info.
