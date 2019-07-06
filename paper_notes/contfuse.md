# [ContFuse: Deep Continuous Fusion for Multi-Sensor 3D Object Detection](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper.pdf) 

_June 2019_

tl;dr: Uses [parametric cont conv](parametric_cont_conv.md) to fuse camera features with point cloud features. It projects camera into BEV space.

#### Overall impression
ContFuse finds the corresponding point in camera images for each of the point cloud (and their KNN), then concats the interpolated camera features to each point. This leads to better 3D object detection results. 

However the ablation tested results show that only one nearest neighbor is needed to achieve best performance, which is surprising and make me doubt the effectiveness of the proposed method (The idea is good but the engineering details may be improved).

Improved by [MMF](mmf.md) (CVPR 2019) from the same group (Uber ATG), which uses multi-task to boost the performance even further. 

This method project camera points into BEV. In this sense it is related to, and perhaps inspired [pseudo-lidar](pseudo_lidar.md) and [pseudo-lidar++](pseudo_lidar++.md). 

#### Key ideas
- Newly proposed method on learning on point clouds can be sorted into two classes: new conv operator (such as parametric cont conv) or pooling op (pointnet, pointnet++).
- ContFuse project image features into BEV and fuse them with the conv layers of a lidar based detector. 
- ContFuse layer:
	- Project lidar points to BEV and perform KNN search (this is $O(N)$, as opposed to the $O(N\log N)$ K-D tree method in the parametric cont conv paper). 
	- Unproject the KNN to 3D and project to camera view. Bilinear interpolation to get each of the points. Get feature map of size $K\times (D_i+3)$, where $D_i$ is the camera feat dimension at each point. 
	- MLP is used to interpolate the features from KNN and summed to the lidar features.


#### Technical details
- The backbone is ResNet 18, meaning that the main contribution comes from the sensor fusion part.
- The BEV detection head is YOLO based, with two orientation and fixed size anchors (one of the advantages of BEV is the scale invariance). The loss are also very straightforward (smooth L1 for all, relative diff for xyz, and log diff of whl, and diff of theta).
- Hard negative mining (evaluate all, but only backprop few).
- During data augmentation, 3D projection matrix is also changed accordingly.

#### Notes
- Find the dev-kit of KITTI. How to evaluate difficulty? (according to height of 2D bbox, occlusion levels and truncation levels)

