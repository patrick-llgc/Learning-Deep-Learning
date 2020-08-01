# [monodepth: Unsupervised Monocular Depth Estimation with Left-Right Consistency](https://arxiv.org/abs/1609.03677)

_July 2019_

tl;dr: Seminal paper on monocular depth estimation with stereo

#### Overall impression
This paper is one pioneering work on monocular depth estimation with self-supervision. 

When people are talking about monocular depth estimation, they mean "monocular at inference". The system can still rely on other supervision at training, either explicit supervision by dense depth map GT or with self-supervision via consistency.

I feel that for self-supervised method there are tons of tricks and know-hows about tuning the model, cf. [google AI's depth in the wild paper](learnk.md).

Monodepth requires synchronized and rectified image pairs. It also does not handle occlusion in training. It is superseded by [monodepth2](monodepth2.md), which focuses on depth estimation from monocular video.

#### Key ideas
- How to make image reconstruction fully differentiable? Use bilinear interpolation, such as in RoIAlign, monodepth and ContFuse.
- Three loss terms
	- Appearance loss $L_{ap}$: photometric image reconstruction loss, weighted average of SSIM and L1 loss
	- Disparity smooth loss $L_{ds}$: L1 regularization of disparity gradient. Edge aware weighting by $e^{-|\partial I|}$
	- Left-right disparity consistency loss $L_{lr}$
- Occlusion/disocclusion is only handled during test time with post-processing with flipped image and average (this selective pp does not work for video based methods). The authors acknowledged that monodepth lacks proper handling of occlusion/disocclusion during training.
- The trained model can also generalize to unseen dataset.

#### Technical details
- Replace deconv with nearest neighbor upsampling followed by a conv per [source](https://distill.pub/2016/deconv-checkerboard/). Note that bilinear upsampling does not work as well as nearest neighbors.
- Did not use batchnorm as it does not improve performance. 
- Augmentation: [0.8, 1.2] for gamma, [0.5, 2.0] for brightness and [0.8, 1.2] for color shifts.

#### Notes
- Read [codebase](https://github.com/mrharicot/monodepth) for STN sampler and SSIM definition.
- What backbone? It seems to be smaller than ResNet 50.