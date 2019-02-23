# [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf)

_Feb 2019_

tl;dr: First model to consume orderless list of point (point cloud raw data) for classification and semantic segmentation.

#### Overall impression
This paper is brilliant in creating a model that directly learns from orderless set. The use of symmetric function of max-pooling is simple, effective and leads to robust performance against outlier and missing data.

#### Key ideas
- Model need to be invariant to N! permutation, and thus is a symmetric set function
- PointNet processes each point individually and identically, and then feed the results to a symmetric function. The resulting model is intrinsically symmetric.
$$
f(x_1, \ldots, x_n) = \gamma \odot g(h(x_1), \ldots, h(x_n))
$$
In the implementation of PointNet, h is a MLP that embed D-dim input into higher dimensional space, g is max-pooling (symmetric), and $\gamma$ is another MLP to process the aggregated feature by g. 
- Concatenate global and local features for **semantic segmentation** of each point.
- PointNet is much more robust to missing data, due to the **max-pooling** operator. Only the **critical point sets** contribute to the max-pooled global features. They capture the object contours and skeletons.

#### Technical details
- Input: Point cloud in the form of 2D matrix, N x D. N is the number of orderless points, and D is the number of channels (x, y, z coord and other features such as RGB, intensity and surface normals).
- It is shown that PointNet can approximate any continuous set function with arbitrary precision, given enough complexity in h (num of dim of embedding).
- Joint alignment network is like a small STN that maps input and features into canonicalized space. This improves the performance of PointNet.


#### Notes
- STN: [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf) <kbd>NIPS 2015</kbd>
