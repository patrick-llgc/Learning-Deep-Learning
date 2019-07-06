# [Deep Parametric Continuous Convolutional Neural Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Deep_Parametric_Continuous_CVPR_2018_paper.pdf)

_July 2019_

tl;dr: Generalization of 2d conv to point cloud. Similar idea to [PointNet](pointnet.md) and [PointNet++](pointnet++.md), and [Point CNN](point_cnn.md).

#### Overall impression
From Raquel's group. This paper is similar to pointNet in that it aggregates and propagates information directly on point cloud. However one of the advantage is that **the output may be different from the input**. So it could be a fancy interpolation method. 

Cons: The math is not rigorous at all. Especially for Fig. 2, the dimensions in the architecture is completely wrong and misleading.

The method is used in [ContFuse](contfuse.md) paper to merge features from camera and lidar. ContFuse used MLP to output results directly instead of modeling the weights to sum over features. 

#### Key ideas
- Use MLP to approximate the point cloud kernel function with full continuous support domain.
- The kernel function gives the weighting given the relative location in the support domain. 
- Arch for one layer (note that dimensions in Fig. 2 is completely off)
	- input: Mx3 desired output coordinates, input points coordinates Nx3, input features NxF
	- find KNN: MxKx3 and support features MxKxF
	- **MLP maps MxKx3 to get MxKxFxO (decomposed into FxO and MxKxO)**
	- get support point weights: MxKxFxO
	- weighted sum of MxKxF and MxKxFxO
	- output: MxO
- Multiple layers can be stacked together, if M=N, for segmentation tasks. 

#### Technical details
- Voxelizing 3D space is memory and computation inefficient. 
- Need K-D tree to find nearest neighbors. This takes almost half of the inference time (16 FPS). 

#### Notes
- There is no code with this paper, which makes it less useful. However it builds the foundation of [Continous Fuse](contfuse.md) and [Multi-task multi-sensor fusion](mmf.md) papers.
- Read [Point CNN](point_cnn.md) paper to see the difference.

