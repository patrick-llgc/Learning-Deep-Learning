# [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/pdf/1706.02413.pdf)

_Feb 2019_

tl;dr: Use multistage PointNet to enable hierarchical learning.

#### Overall impression
The paper is an excellent piece of scientific writing full of insightul and concise statements. The paper extends the work of PointNet from multi-stage/hierarchical viewpoint. PointNet++ applies PointNet recursively on a nested partioning of the input data. It is impressive to see the authors uses multi-scale technique to address the loss of robustness in vanilla PointNet++.

#### Key ideas
- The original PointNet does not capture local features as the features of each point is extracted and aggregated into global signature directly (though a single maxpooling operation). This is like using a fully connected layer. This paper introduced hierarchical learning of features, resembling the learning process of CNNs.

  > The ability to abstract local patterns along the hierarchy allow better generalizability to unseen cases.

- The set of points are partitioned into overlapping local regions (by the distance metric of the underlyinbg space). And each local region is processed by PointNet to extract local features. 

  > PointNet is effective in processing an unordered set of points for semantic feature extraction.

- The data partitioning is done with farthest point sampling (FPS). The receptive field depends on the input data and the metric defined in the underlying Euclidean space.


- The original PointNet does not have the issue of varying sampling densities. To address this issue, PointNet++ combines features from multiple scales to increase rbustness and detail capture.

#### Technical details
- Math formulation: PointNet or PointNet++ learns set functions f that take a descrete metric space $\mathcal{X} = (M, d)$ as the input and produce semantic information regarding $\mathcal{X}$. The set funciton f can be calssificaiton or sementic segmentation.
- PointNet++ has a series of **set abstraction** levels. Each set abstraction level has Sampling Layer, Grouping Layer and PointNet Layer. 
  - Sampling Layer: Use iterative fartherst point sampling (FPS) to find centroid points. $N \times (d+C) \rightarrow N' \times d $, where N' is the number of centroids
  - Grouping Layer: $N \times (d+C) \rightarrow N' \times K \times (d +C)​$. Note that K varies for different centroid. 
    - Manhattan distance in CNN vs metric distance in PointNet. Ball query finds all points within a certain distance to a given centroid. Comared with kNN, ball query guarantees a fixed region scale and thus is more robust and general.
  - PointNet Layer: convert flexible number of points into a fixed length local region feature vector. $N' \times K \times (d +C) \rightarrow N' \times (d+C')$. **Note that the features extracted from the centroids neighborhoods are assigned to the centroids.** 
    - The coordinates of points in a local region are firstly translated into a local frame relative to the centroid point. This allows the PointNet to learn more generalized local features.
    - Note that the PointNet Layer in each set abstraction level learns on different scales. The first PointNet Layer learns more about local features and later PointNet Layers learns more global features.
- Robust feature learning through multi-scale and multi-resolution
  - MSG: multi-scale grouping calculates features with multiple radius. This has the best performance. 
  - MRG: multi-resolution grouping not only uses the features calculated with the point set results from the last layer, but also directly on all points in the local region. This allows learning of differnet weighting strategies for different sampling densities. MRG is less computationally intensive as the second half of the computation can be shared by all regions.
- Point feature propagation with semantic segmentaiton
  - For semantic segmentaion, we can sample all points as centroids (without subsampling), but this is too costly.
  - Upsampling features from sampled data, and concatenate with previous level. The concatenated features are passed through a **uint PointNet** (similar to a 1x1 conv in CNN) to adjust the feature size. This is quite like the U-Net or FPN structure.
  - The upsampling used inverse distance weighted average, as compared to bilinear upsampling or transposed convolution in U-Net. **Can we propose a transposed PointNet layer** to learn the best upsampling strategy rather than hand-craft inverse distance weighted average?
- Ablation results show that DP (random input dropout during training) is a very effective stategy. MRG and MSG helps but only slightly.
#### Notes
- Point cloud can be complete or partial scan (from only one view point, generally the case for autonomous driving).
- The work is reviewed by the author Charles Qi in [this video](https://www.youtube.com/watch?v=Ew24Rac8eYE).
- Maybe we can introduce **more layers** into the structure? In a sense, PointNet++ achieves convolutional behavior via applying fully connected layers in a convolutional fashion. We could use a naive basic operator similar to 3x3 conv and leverage the existing CNN structures.

  - The authors actually did experiment and found out too small a neighborhood is bad. PointNet may need a large enough neighborhood to capture useful information.

- This paper introduces many concepts with the help of metric space. Metric space: **把metric function或distance function的共性提取出来，在一个任意的集合上规定：**

  **度量空间是一个有序对，记作(X,d)，其中 X 是一个集合，d是 X 上的metric function:** $X \times X \to [0,\infty) $ ，它把 X 中的每一对点 x，y 映射到一个非负实数，**并且满足如下四条公理：**

  1. 非负性： $d(x,y)\geq 0$
  2. 唯一性： $d(x,y)=0\Leftrightarrow x=y$
  3. 对称性： $d(x,y)=d(y,x)$
  4. 三角不等式： $ x,y,z\in X$$ ，$$d(x,z)\leq d(x,y)+d(y,z)$