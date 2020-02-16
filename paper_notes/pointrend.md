# [PointRend: Image Segmentation as Rendering](https://arxiv.org/abs/1912.08193)

_January 2020_

tl;dr: Find most uncertain points in segmentation and use both coarse RoI feature map and fine feature map to predict results. 

#### Overall impression
The paper tells a great story about borrowing ideas from rendering to segmentation. However the idea of coarse-to-fine has been explored extensively before. The main novelty of this paper is how to save computation by non-uniform sampling.

> A regular grid will invariably oversample the smooth areas while simultaneously undersample object boundaries. For semantic segmentation, we use feature map of 1/8 size of input. Or 28x28 for instance segmentation.
 
PointRend is a module that can be incorporated in instance/semantic segmentation frameworks to improve results. 

#### Key ideas
- PointRend has 3 main components: 1) point selection strategy; 2) point-wise feature representation 3) point head to predict a label.
- Sampling strategy varies from training to inference.
- Inference: **iterative** process.
	- Bilinear upsample prediction 
	- Find most uncertain N points (with prob ~0.5)
	- Bilinear sample from fine feature map (FPN-P2) and coarse feature map (7x7 Mask RCNN-like head)
	- MLP based on concatenated features to predict K-classes.
- Training
	- Over generation: generates KN (K>1). K = 3
	- Importance sampling: pick bN (b<1) most uncertain points. b = 0.75
	- Coverage: uniform sample (1-b) N for the rest of the points. 

#### Technical details
- This method is reminiscent of Hypercolumn to improve semantic segmentation.

#### Notes
- [detectron2](https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend/point_rend)
- [Hypercolumns for Object Segmentation and Fine-grained Localization](https://arxiv.org/abs/1411.5752) <kbd>CVPR 2015</kbd>
	- Hypercolumn is very similar to and inspired FPN directly.


