# [Learning to Find Good Correspondences](https://arxiv.org/abs/1711.05971)

_March 2020_

tl;dr: Given the coordinates of matching candidate, assign weight to the candidates.

#### Overall impression
The paper is inspired by [PointNet](pointnet.md), and recycles the permutation-invariant property of PointNet to assign weight to correspondence candidates.

The paper uses a hybrid approach of combining classification and estimation of Essential Matrix together. However, even only with classification the performance is not as bad. 

Another interesting thing is that the paper uses only the keypoint position of (x1, y1, x2, y2) x Batch_N as input to the PointNet-like neural network and completely **discard the descriptors**.

This paper inspired [NG-RANSAC](ng_ransac.md) and [KP2D](kp2d.md).

#### Key ideas
- Differentiable way to estimate Essential Matrix. This required a closed form solution.
- **Context Normalization** among N instances (similar to instance normalization). This will introduce global information implicitly, as compared to the direct concatenation of global information to local information.
	- This paper shows that using context normalization actually perform better than concatenate global information
- The differentiable formulation of Essential Matrix estimation just provides an effective supervision signal. At inference time, this step is skipped. 
- During inference, RANSAC is still used after the first round of filtering, then pass on to conventional 8-point algorithm to find Essential Matrix.
- It is much faster than RANSAC. It runs at 13 ms on GPU with 2k candidates, whereas RANSAC will need 373 ms.

#### Technical details
- Summary of technical details

#### Notes
- "Correspondence" is essentially "Association". This is one good direction

