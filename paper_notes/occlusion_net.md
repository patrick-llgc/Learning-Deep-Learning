# [Occlusion-Net: 2D/3D Occluded Keypoint Localization Using Graph Networks](http://www.cs.cmu.edu/~mvo/index_files/Papers/ONet_19.pdf)

_September 2019_

tl;dr: Using GNN to reason position of occluded points using multi-view geometry (trifocal tensor),

#### Overall impression
Occlusion-Net's builds on the strong baseline of keypoint detection of Mask RCNN and is better in two ways: it can reason the location of the occluded points better, and it classifies whether a point is occluded much more accurately.

The dataset [CarFusion](http://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion/cvpr2018/index.html) should have the same car annotated in different view points and the correspondence is annotated too. Both visible and occluded keypoints are annotated but only visible keypoints are used for training -- the occluded ones are used for evaluation only.

#### Key ideas
- Three steps:
	- visible 2D keypoint regression with Mask RCNN (inspired by stacked hourglass)
	- Reason which edges are occluded
	- Reason which points are occluded --> not sure why cannot the two steps be combined?
- The uncertainty of visible and occluded keypoints overlaps significantly, and thus cannot be used for classification. --> perhaps can add additional bit for classification.
- provide supervision for a hidden point with two views where the point can be seen and train a **trifocal tensor**.
- Evaluation with PCK (Probability of Correct Keypoint): if pred and gt are within $\alpha L$ then it is counted as TF, where L= max(w, h). $\alpha$ is usually set to be 0.1. --> occlusion net has PCK of 0.8-0.9 @ $\alpha$=0.1.

#### Technical details
- The trifocal tensor extends the fundamental matrix in rectification to multiview (>=3) geometry. It is a 3x3x3 matrix, and projects to 3 3x3 matrix to correlate the 3 pairs. 
- The dataset has 50k images and 100k annotated cars. 
- Predicting occluded keypoints as heatmaps generates large errors. 

#### Notes
- The model performed pretty well on visible keypoints, and can be used to generate pseudo labels. 
- Mask RCNN and Stacked Hourglass are pretty much the SOTA for keypoint regression.
- Mask RCNN need to be modified to include additional bit for existence regression. This should be a fairer comparison with occlusion net.
- In deployment, occlusion net does not perform well for classes with truncated cars. However in the paper it performs pretty well. Maybe we need padding to solve this? Do all points need to be inside bboxes?
