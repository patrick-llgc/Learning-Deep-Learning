# [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)

_January 2020_

tl;dr: Learn a real-time **feature detector** based on self generated data.

#### Overall impression
A **[local feature](https://dsp.stackexchange.com/questions/24346/difference-between-feature-detector-and-descriptor)** consists of an interest point (key point, salient point) and a descriptor. Multiview geometry studies how to recover the transformation between the views and infer the 3D positions of these key points, based on the assumption that the points are **matched** across multiview images. How do we learn a feature detector?

Now from the stand point of generative modeling, if we know the key points of one image, we can do homographic transformation of the image together with the key points. This will generate tons of training data to learn descriptor. (Yet another example of transferring knowledge of a well-defined math problem to neural nets)

How to learn detector in the first place? We can render 2D projections with 3D objects with known interest points. From synthetic to real images, in order to bridge the sim2real gap, test time augmentation is used to accumulate interest point features. This TTA is called "homographic adaptation".

The above three steps largely summarizes the main idea of this paper: 

- MagicPoint: pretraining on synthetic data: (key point detector)
- Homographic Adaptation: TTA on real images
- SuperPoint: MagicPoint with descriptors trained with image pairs undergoing known homographic transformation. The descriptor is used for image matching tasks.

The design of catch-all channel dustbin to recalibrate softmax heatmap is interesting, and both [SuperPoint](super_point.md) and [VPGNet](vpgnet.md) used the same trick.

From the point that the keypoint detection and representation are shared across two tasks, SuperPoint is similar to [associative embedding](associative_embedding.md).

This paper also inspired [unsuperpoint](unsuperpoint.md) which does not require pseudo-GT to train.
 
#### Key ideas
- Detecting human body keypoints is semantically well defined, but detecting salient points in an image is semantically ill-defined. Hard to manually label these points.
- Architecture:
	- VGG style backbone, downsample by 8x
	- Keypoint detection branch has 8x8+1=65 channels. The additional channel is a "no-interest point" catch-all **dustbin**. It is helpful to recalibrate the keypoint heatmap. The whole flow goes like: 65 ch --> Softmax --> drop dustbin channel --> 64 ch --> reshape to 8x upsampled image. 
	- Descriptor: 256-dim vector for 8x8 downsmpled feature map. Bi-cubic interpolation (to accommodate subpixel locations) and then L2 normed. Semi-dense descriptor saves training memory and keeps run time tractable. --> This is different from [unsuperpoint](unsuperpoint.md) which include interpolation inside the network.
- Loss function: 
	- point loss: pixel wise cross entropy on original scale
	- descriptor loss: hinge loss with positive margin 1 and negative margin 0.2 (zero loss as long as cosine similarity is below 0.2 for non-matching pairs and as high as possible for matching pairs) --> there are O(n^2) terms in this loss! 
	- cf [associative embedding](associative_embedding.md) on how to avoid calculating large numbers of items in loss in a grouped setting. Not applicable in a general setting.


#### Technical details
- Magic points trained on synthetic images are more robust compared to classical detectors in the presence of noise (thanks to conv net).
- NMS for detector: NMS within a radius of 3 pixels. This helps control the density of keypoints.
- Homographic adaptation: tta with 100 samples. Diminishing return above this number.
- Random homographic crop for training descriptor: essentially random crop a 4-sided polygon from the orignal image and warp it to rectangle.

	> Homographies give exact or almost exact image-to-image transformations for camera motion with only rotation around the camera center, scenes with large distances to objects, and planar scenes.

#### Notes
- A good key point detector should have good [repeatability and accuracy](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/vibes_ijcv2004.pdf):
	- the **repeatability**, i.e., the average number of corresponding regions detected in images under different geometric and photometric transformations, both in absolute and relative terms (i.e., percentage-wise)
	- the **accuracy** of localization and region estimation. H is the homography relating the two images.
	$$
	1 − \frac{R_{µa} \bigcap  R_{H^T µb H}}{R_{µa} \bigcup R_{H^T µb H} } < O
	$$
	- MLE: mean localization error (in pixels)
	- NN mAP: nearest neighbor mAP (measures how discriminative the description is, by varying matching threshold)	
- In an [earlier work](https://arxiv.org/abs/1707.07410), MagicWarp is used to regress the homographic transform directly and use re-projection error as loss.