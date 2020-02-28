# [FixRes: Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423) 

_February 2020_

tl;dr: Conventional imageNet classification has a train/test resolution discrepancy (domain shift).

#### Overall impression
Scale invariance/equivariance is not guaranteed in CNN (only shift invariance). The same model with different test time input will yield very different statistics. The distribution of activation changes at test time, the values are not in the range that the final cls layers were trained for.

In ImageNet training, conventional way is to use 10-time crop (center, four corners, and their mirrors) and test time is always central crop. This leads to a discrepancy of the statistics in training/test.

Simple solution: **finetune last layer** with test time scale and resolution, as the final stage of training.


#### Key ideas

#### Technical details
- Larger test crops yields better results.
- A similar work is MultiGrain, where the p-pooling is adjusted to match the train/test-time stats.
- GeM (generalized mean pooling) p-pooling: a generalization of average pooling and max pooling
	- cf LSE pooling in [From Image-level to Pixel-level Labeling with Convolutional Networks](https://arxiv.org/abs/1411.6228) CVPR 2015
	- Image/instance retrieval requires adjusting p-pooling for better accuracy

#### Notes
- Questions and notes on how to improve/revise the current work  

