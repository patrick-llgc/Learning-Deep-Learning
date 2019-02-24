# [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf)

_Feb 2019_

tl;dr: First model to consume orderless list of point (point cloud raw data) for classification and semantic segmentation.

#### Overall impression
This paper is brilliant in creating a model that directly learns from orderless set. The use of symmetric function of max-pooling is simple, effective and leads to robust performance against outlier and missing data. The theoretical explanation and visualization of this paper is superb. This is perhaps one of the thought-provoking papers I have ever read.

#### Key ideas
- A point cloud is a point sets in $\mathbb{R}^n$, where n is the dimension.
	- Unordered: Model need to be invariant to N! permutation, and thus is a symmetric set function
	- Invariance under transformation: for rotation and translation of all points all together should not modify the global classification and point segmentation.
- Compared with voxelized input, directly learning on point cloud input is time and space efficient. Volumetric representation is contrained by its resolution due to data sparsity and computation cost of 3D resolution.
- PointNet processes each point individually and identically, and then feed the results to a symmetric function. The resulting model is intrinsically symmetric.
$$
f(x_1, \ldots, x_n) = \gamma \odot g(h(x_1), \ldots, h(x_n))
$$
In the implementation of PointNet, h is a MLP that embed D-dim input into higher dimensional space, g is max-pooling (symmetric), and $\gamma$ is another MLP to process the aggregated feature by g. 
- For **semantic segmentation**, the global and local features are concatenated for classification of each point. The output is a $n \times m$ scores, for each n points and m semantic categories. This semantic segmentation can be post-processed with connected component analysis and used for **3D object detection**.
- PointNet is much more robust to missing data, due to the **max-pooling** operator. Only the **critical point sets** contribute to the max-pooled global features. They capture the object contours and skeletons. Point sets between the critical point sets and upper bound shapes generates exactly the same global shape features. 
- Computationally, PointNet scales linearly with input size. Multi-view 2D CNN scales quadratically with spatial resolution and 3D CNN on volumetric representation scales cubically with volume size.


#### Technical details
- Input: Point cloud in the form of 2D matrix, N x D. N is the number of orderless points, and D is the number of channels (x, y, z coord and other features such as RGB, intensity and surface normals).
- It is shown that PointNet can approximate any continuous set function with arbitrary precision, given enough complexity in h (num of dim of embedding).
- Joint alignment network is like a small STN that maps input and features into canonicalized space. This improves the performance of PointNet.
- Additional loss on transformation matrix to enforce it is orthogonal transformation $L_{reg} = ||I - AA^T||^2$.
- Augmentation on the fly: rotation, Gaussian noise jitter.
- The bottleneck dimension (channels, or num of filters) and number of input points both affect results. Even with 64 points (furthest point sampling) we can achieve descent performance (83% cls accuracy, vs87% @ 1K points).

#### Notes
- STN: [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf) <kbd>NIPS 2015</kbd>
- [Blensor](https://www.blensor.org/) is a simulator to generate lidar and kinect point cloud.
- PointNet can approximate any continuous set function to any precision. 
- Hausdorff Distance measures how far two subsets of a metric space are from each other.
- Sup vs max: Supremum (least upper bound) needs not be attained while maximum is. When maximum exist, maximum=supremum.