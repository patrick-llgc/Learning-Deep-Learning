# [SWAHR: Rethinking the Heatmap Regression for Bottom-up Human Pose Estimation](https://arxiv.org/abs/2012.15175)

_April 2021_

tl;dr: Automatically adapt gaussian kernel size for keypoint detection to accommodate scale variation and labeling ambiguity.

#### Overall impression
The paper addresses one key issue in bottom up key point detection.

#### Key ideas
- SAHR (scale adaptive heatmap regression) predict an additional **scale maps s** to modify the standard deviation to be $\sigma_0 s$. This improves the baseline of HPE (human pose estimation) which covers all keypoints by Gaussian kernels with the same stdev.
- Loss: L2 heatmap regression loss + regularization loss
	- The size map can also be interpreted as **uncertainty**. 
	- Regularizer loss has to be added to stablize training. This is more like the [aleatoric uncertainty estimation](kl_loss.md).
- WAHR (weight adaptive heatmap regression): more like **focal loss** used by [CornerNet](cornernet.md) and [CenterNet](centernet.md). It automatically downweighs the loss of easier samples.
	- Suppose P is prediction, H is GT heatmap
	- original l2 loss $||P-H||_2^2$
	- new loss $W ||P-H||_2^2$, and $W = H^\gamma ||1-P|| + (1-H^\gamma) ||P||$. 
	- Intuitively, for points larger than a certain thresh, the weight is more like $||1-P||$ and penalizes more if the prediction is small. 

#### Technical details
- **Bottom-up vs top-down** approaches for human keypoint detection. 
	- Bottom up approaches could be faster as the running speed is not limited by the number of instances, and is not limited by the performance of the object detector. See [PifPaf](pifpaf.md). Keypoints are grouped by [Associative Embedding](associative_embedding.md).
	- Top down approaches are two-stage methods. Images are cropped out, resized before feeding into a single-human keypoint detector. The inference time scales with number of people in the image. Currently top-down approaches still have better performance, **except in crowded scenes**. 
- Each keypoint has a scale map s, to pair with the heatmap.
- Ablation study
	- Naive implementation is to estimate size of heatmap from bbox size. This is actually worse than the baseline performance with a uniform stdev.

#### Notes
- Code on [github](https://github.com/greatlog/SWAHR-HumanPose)
- Q: Can we apply this idea to anchor free object detection? And use the stdev to do adaptive NMS?