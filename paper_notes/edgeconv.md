# - [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/pdf/1801.07829.pdf) (EdgeConv, DGCNN)

_Mar 2019_

tl;dr: Extact semantic features from point cloud by iteratively performing convolution on a dynamically updated neighborhood.

#### Overall impression
This paper extends on the PointNet architecture. This paper addresses the same probelm that [pointNet++](pointnet++.md) tried to solve: PointNet treats each point input independently, and there is no local neighborhood information used. Instead of using farthest point sampling, EdgeConv uses kNN. 

#### Key ideas
- EdgeConv (DGCNN) dynamically updates the graph. That means the kNN is not fixed. Proximity in feature space differs from proximity in the input, leading to nonclocal diffusion of information throughout the point cloud.
	- Dynamic update of the graph makes sense, but ablation test shows it only gives minor improvement.
- EdgeConv operation transforms an F-dimensinoal point cloud with n points to F'-dim point cloud with n points.
$ x'_i = \sum_{j:(i, j)\in E} h_{\theta}(x_i, x_j) $
Note that the sum sign is a placeholder and can be max operation. 
	- If $h(x_i, x_j)=\theta_j x_j$, then this is conventional convolution in Euclidean space.
	- If $h(x_i, x_j) = x_i$, then this is point net.
	- In this paper $h(x_i, x_j) = h(x_i, x_j-x_i)$, which captures the global information (x_i) and the local information (x_j - x_i)


#### Technical details
- Point cloud is flexible and scalable geometric representation. Processing point cloud data direclty bypasses expensive mesh reconstruction or denoising. Project point cloud into 3D grid introduces quantization artifacts and excessive memory use.
- PointNet is senseitive to the global transformaton of the point cloud, and thus uses a t-net to transform the point cloud into a cannonical viewpoint. 

#### Notes
- Q: How are the nearest neightbors identified? Using what metrics? (In other words, how is the directed graph G containing kNN constructed?)
	- A: An adjacency matrix is calculated and top k points with the smallest distance is found for each row (distance to all points for a given point). See [official implementation](https://github.com/WangYueFt/dgcnn/blob/master/models/dgcnn.py#L27).
- Q: Why the spatial transformer network has the input of nxkx6 dimension? 
	- A: 6 = 3 (point coordinates) + 3 (difference of coordinates between point with its kNN).
- Intrinsic vs extrinsic descriptors: 
	- Extrinsic descriptors are derived from the coordinates of the shape
	- Intrinsic descriptors treat the 3D shape as a manifold and is invariant to isometric deformaton (different poses of human body, etc)
![](https://shapeofdata.files.wordpress.com/2013/08/inextrinsic.png?w=640&h=244)

> If you were somehow shrunk down so that you could walk around “inside” the data set and could only see nearby data points, you might think of the two blobs in each data set as rooms, and the narrow neck as a hallway. If you walked from one room to the other, you might not notice whether or not it was curving. So as in the building example, you would have a hard time telling the difference between the two sets from “inside” of them. ([source](https://shapeofdata.wordpress.com/2013/08/06/intrinsic-vs-extrinsic-structure/), [source2](https://shapeofdata.wordpress.com/2013/08/06/intrinsic-vs-extrinsic-structure/))
