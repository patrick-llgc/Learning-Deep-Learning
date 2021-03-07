# [DSNT: Numerical Coordinate Regression with Convolutional Neural Networks](https://arxiv.org/abs/1801.07372)

_March 2021_

tl;dr: A differentiable way to transform spatial heatmaps to numerical coordinates.

#### Overall impression
The SOTA method for coordinate prediction is still heatmap-based keypoint regression (heatmap matching) instead of direct numerical coordinate regression. 

Previous method obtains numerical coordinates by applying armgax to the heatmaps, which is a non-differentiable operation. Gradient flow starts from heatmap and not the numerical coordinates. The handcrafted features in designing heatmap and the postprocess to obtain numerical coordinates leads to sub-optimal system design.

DSNT proposed a way to back-propagate all the way from the predicted numerical coordinates tot he input image.

DSNT can only handle one keypoint per heatmap. How to extend the work to multiple keypoints per heatmap is still open to research. --> Maybe try to impose a neighborhood.

#### Key ideas
- Applying a [CoordConv](coord_conv.md) layer and perform inner product of the heatmap with X and Y coordinate maps. This essentially uses mean (differentiable) to find the mode (non-differentiable) of the heatmap.
- Regularization to encourage the heatmap blobs to resemble a gaussian shape.

#### Technical details
- An alternative way is by using a soft-argmax ([Human pose regression by combining indirect part detection and contextual information]())

#### Notes
- Questions and notes on how to improve/revise the current work  

