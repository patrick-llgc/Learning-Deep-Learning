# [CornerNet: Detecting Objects as Paired Keypoints](https://arxiv.org/pdf/1808.01244.pdf)

_April 2019_

tl;dr: Detect the top-left and bottom-right corner of the bbox, and learn an encoding for data association ([associative embedding](associative_embedding.md)). It outperforms even multi-stage detectors such as mask rcnn and cascade rcnn.

#### Overall impression
The paper is the first anchor-less object detection paper in 2018 and rekindled people's interest in anchor-less object detection framework. The corner pooling operation seems to have draw inspiration from the way humans draw bboxes. I would argue that the existence of corner pooling itself act as a proof that **bbox is a bad representation for object detection**.

#### Key ideas
- Anchor based methods usually used a large number of anchor boxes (RetinaNet uses 100k). This leads to huge imbalance in the classification problem. Thus we use focal loss.
- CornerNet uses the same backbone to predict a heatmap that represents the top left corner, a heatmap that represents the bottom right corner, and an embedding vector for each detected vector (and one branch predicting the offsets). 
- **[Associative embedding](associative_embedding.md)**: The network predicts similar embeddings for corners belonging to the same object. The loss is very similar to the triplet loss. When the averaged embedding from two object is further than a threshold, the loss is set to 0, thus ignored. 
- **Corner pooling**: a corner is often outside an object and there is usually no local visual evidence for the presence of corners. Corner pooling does max pooling of all pixel values on the right and all pixel values to the bottom and sums them. 
- (Inverse) Gaussian mask for regressing one-hot maps. This is an alternative to predicting a Gaussian blurred ground truth. This method is also used in [CSP](csp_center_scale.md).
- CornerNet generates even more high quality bbox (IoU >= 90) than cascade RCNN. This is really surprising, and maybe due to the fact that it regresses to the points directly.

#### Technical details
- CornerNet predicts $O(wh)$ corners to represent $O(w^2h^2)$ bboxes.
- CornerNet uses Stacked Hourglass as backbone for keypoints regression. Ablation test shows it performs better than ResNet. For keypoint regression, we should consider using Stacked Hourglass.
- Detecting corners (keypoints) can be seen as a binary classification task (hit or miss), given a certain error tolerance. 

#### Notes
- The authors argue that detecting corners is easier than detecting center as corners only depend on two sides but center depends on four sides. See [CSP](csp_center_scale.md) and [ExtremeNet](extremenet.md) for comparison.
- [Presentation at ECCV 2018](https://www.youtube.com/watch?v=aJnvTT1-spc)


