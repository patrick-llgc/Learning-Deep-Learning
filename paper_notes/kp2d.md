# [KP2D: Neural Outlier Rejection for Self-Supervised Keypoint Learning](https://arxiv.org/abs/1912.10615)

_March 2020_

tl;dr: Improvement of [unsuperpoint](unsuperpoint.md) with better designed loss and one auxiliary outlier detection task.

#### Overall impression
This paper is inspired by [unsuperpoint](unsuperpoint.md). However it implemented multiple improvement which boosted the performance quite a bit.

Although the proposed method does not achieve best performance all the time, it is **within reasonable margin of the best performing model variant**.

The method achieves SOTA **repeatability** and good performance in other metrics.  

The proxy task of identifying outliers during training is inspired by [NG-RANSAC](ng_ransac.md).

#### Key ideas
- Location loss relaxation
	- instead of encouraging the distribution of the [0, 1] to be uniform, KP2D digs into why it is the distribution not uniform and relaxed the values to beyond [0, 1]. This allows the keypoints to go to neighboring anchors. 
	- This is due to the fact that the point pairs are not injective. One point in the source image can be matched to multiple points in the target image just by the distance threshold.
- Triplet Descriptor loss: Uses triplet loss instead of contrastive loss in [unsuperpoint](unsuperpoint.md).
	- Each keypoint in input image $p_i \in \mathbf{p_s}$ in source image has descriptor $f_i$. Among all warped points $p_i^* \in \mathbf{p_t^*}$ in target image, they have corresponding descriptors $f_i^*$. The positive example $f_{i, +}^*$ is sampled at the warped position, and the negative example $f_{i, -}^*$ is the closest/hardest negative point. 
	- $L_{desc} = \sum_i \max(0, |f_i - f_{i, +}^* |_2 - |f_i - f_{i, -}^*|_2 + m)$
- IO-Net: use outlier rejection as auxiliary task during training only. PoitNet-like architecture. See [NG-RANSAC](ng_ransac.md) and [learning good correspondence](learning_correspondence.md).
	- Input is 5 numbers: $p_s$, $p_t^*$, $|f_s - f_t^*|_2$, and output is whether the point-pair is an "inlier" set. --> Note that the prediction is based on the source and mapped target image, and the corresponding features at those locations.
	- IO-Net loss compares the prediction of IO-Net and the point-pair based on point-pair keypoint location.
	- Only keypoints with 300 keypoint pairs with the lowest scores are used for training

#### Technical details
- Changing backbone from VGG in [KP2D](kp2d) to ResNet18 in [KP3D](kp3d.md) improves performance.

#### Notes
- Triplet loss is a relatively new concept, only proposed in [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) <kbd>CVPR 2015</kbd>.
- Contrastive loss is an older idea. It comes from [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) <kbd>CVPR 2006</kbd>.
- Contrastive loss is two-stream, and Triplet loss is three-stream. [source](http://slazebni.cs.illinois.edu/spring17/lec09_similarity.pdf)
- Neural Ransac proposes to classify whether each pair is an inlier set, heavily inspired by [NG-Ransac](ng_ransac.md). This can be useful for radar-camera deep association.