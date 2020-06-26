# [PackNet-SG: Semantically-Guided Representation Learning for Self-Supervised Monocular Depth](https://arxiv.org/abs/2002.12319)

_May 2020_

tl;dr: Use pretrained semantic segmentation model to guide [PackNet](packnet.md). Use two rounds of training to solve **infinite depth** issue.

#### Overall impression
This paper has a different focus than other previous methods which focuses on accurate VO. This paper focuses on accurate depth estimation, especially dynamic object like cars. 

Based on the framework of [SfM-Learner](sfm_learner.md) and SfM, cars moving at the same speed as ego car will be projected to infinite depth. To avoid the infinite depth issue, a simple way to do this is to mask out all dynamic objects and train SfM on static scenes. But this will not give accurate depth on cars during inference. 

So the best way to do this is to train SfM on static scenes (in parking lot, e.g.) and during inference, the depth network will generalize to moving cars, as the depth network only takes in a single image. 

The infinite depth issue is also tackled in [Struct2depth](struct2depth.md).

#### Key ideas
- The paper uses a pretrained semantic segmentation network and [Pixel adaptive convolution](https://arxiv.org/abs/1904.05373) to perform content adaptive upsampling.
- Two-pass training. First pass will train a model with infinite depth issue, which is used to resample the dataset by automatically filtering out sequences with infinite depth predictions that violate a basic geometric prior. If some pixels's height are **significantly below the ground**, then we filter out the image. This roughly filters out only 5% of training data.
	- Also need to filter out **static images**, similar to [SfM-learner](sfm_learner.md).
- [Pixel adaptive convolution](https://arxiv.org/abs/1904.05373) helps to preserve spatial details, similar to [PackNet](packnet.md) and can be used together.

#### Technical details
- About 40000 images for training.
- Some networks explore semantic information by predicting both from the same network. [Towards Scene Understanding](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Towards_Scene_Understanding_Unsupervised_Monocular_Depth_Estimation_With_Semantic-Aware_Representation_CVPR_2019_paper.pdf) uses a conditional decoder to predict either depth or semantic segmentation, by concatenating one channel to the jointly learned features.

#### Notes
- [Github code](https://github.com/TRI-ML/packnet-sfm/)
- [Pixel Adaptive Convolutional Neural Networks](https://arxiv.org/abs/1904.05373) <kbd>CVPR 2019</kbd> modifies original conv kernels with another content aware kernel. [Video demo](https://www.youtube.com/watch?v=gsQZbHuR64o)
- [Towards Scene Understanding: Unsupervised Monocular Depth Estimation With Semantic-Aware Representation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Towards_Scene_Understanding_Unsupervised_Monocular_Depth_Estimation_With_Semantic-Aware_Representation_CVPR_2019_paper.pdf) <kbd>CVPR 2019</kbd>

