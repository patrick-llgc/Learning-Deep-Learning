# [UnsuperPoint: End-to-end Unsupervised Interest Point Detector and Descriptor](https://arxiv.org/abs/1907.04011)

_March 2020_

tl;dr: Train an interesting point detector without pseudo-gt.

#### Overall impression
Most previous DL-based interest point detection works focuses on the descriptor part, but still rely on traditional keypoint detection (such as FAST). The main challenge is that the keypoint in natural image is hard to define.

This paper is inspired by [SuperPoint](superpoint.md). However superpoint requires pseudo-gt generated from synthetic images followed by homographic adapttion (extensive, ~100 TTA to increase recall). Interesting/salient points emerges naturally after training.

This paper has tons of losses! Balancing them is quite a task.

#### Key ideas
- Multitask training
	- coarse score heatmap (x8 downsampled): no need to do NMS, and encourages interest points to be more homogenously distributed
	- subpixel location regression
	- Descriptor description. Using subpixel location for interpolation
- Distinction between superPoint
	- There is no direct supervision of interest point position. No tedious pretraining on synthetic data and homographic adaptation
	- Direct regression instead of channel wise classification for subpixel location prediction
	- Interpolation happens inside network. Superpoint does interpolation after inference
- Training using Siamese network, using known homography transformation to regularize training
- Loss --> Tons of losses!
	- Location loss for point-pair
	- Score loss for point-pair
	- unsupervised loss: match may not be injective. Iterate through all pairs and find close enough pairs. Position loss + score loss + matching loss. 
		- the matching loss is $\frac{s_k^a + s_k^b}{2} (d_k - \bar{d})$. When $d < \bar{d}$, the score has to be set higher. This lets the network to output high scores for points which the network believes can be retrieved reliably under the homographic transformation.
	- **uniform distribution** loss: encourage distribution inside [0, 1] to be uniform. 
		- Sorting function is differentiable.
	- descriptor loss: hinge loss, same as [superpoint](superpoint.md) O(N^2).
	- de-correlation loss: the dimensions between descriptors are encouraged to be independent. Minimizing the L2 norms of the off-diagonal elements in the correlation matrices (See [L2-Net in ICCV 2017](http://www.nlpr.ia.ac.cn/fanbin/pub/L2-Net_CVPR17.pdf)).

#### Technical details
- Training with COCO datasets but without labels.

#### Notes
- [知乎](https://www.zhihu.com/question/344670370/answer/899793208)
- Question: what prevents the network from outputing an all-zero score map and essentially setting all losses to zero?
