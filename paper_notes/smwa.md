# [SMWA: On the Over-Smoothing Problem of CNN Based Disparity Estimation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_On_the_Over-Smoothing_Problem_of_CNN_Based_Disparity_Estimation_ICCV_2019_paper.pdf)

_November 2019_

tl;dr: Use single-modal weighted average (SMWA) instead of full-band weighted average to reduce the over-smoothing problem in depth estimation.

#### Overall impression
Long tail is a typical problem in CNN-based depth estimation, for both monocular and disparity based, supervised or unsupervised methods. This point is echoed in [pseudo lidar](pseudo_lidar.md), [pseudo lidar end to end](pseudo_lidar_e2e.md) and [ForeSeE](foresee_mono3dod.md). The pseudo-lidar point with long tails confuses 3d object detectors.

> [DC](depth_coeff.md) focuses on depth completion while [SMWA](smwa.md) focuses on depth estimation from stereo pairs.


This paper seems to be heavily influenced by [DC](depth_coeff.md), including the soft one-hot encoding and cross entropy loss. But it did not acknowledge that.

#### Key ideas
- The proportion of pixels in a modal in all regions and the edge regions are different. The ones in edge regions are more like a bi-modal distribution.
- Single Modal Weighted Average (SWMA) in inference 
	- Find maximum value
	- Identify the left and right range by monotonic descent
	- Weighted average only in that range
- Training: Regression vs cross entropy
	- Changing L1 loss on weighted averaged prediction to cross entropy with soft (gaussian smoothed, in this paper with variance of 2) one-hot label. **CE led to superior results than L1.**
- Evaluation: soft edge error (SEE)
	- For each pixel in the edge region, find the min error between the neighboring prediction and GT. 
	- Average all pixels in the edge region.

#### Technical details
- 3D CNN based method: 
	- construct a 4D tensor based on concatenating left and right feature on location corresponding to specified disparity value, essentially adding a disparity bin dimension. 
	- The 4D tensor goes through 3D CNN and output d/4 x h/4 x w/4 feature map, trilinearly upsampled to full resolution d x h x w. 
	- Per-pixel softmax to get probability for each pixel location. Then d is weighted average of all bins. 
- Minor misalignment of disparity at boundaries is acceptable as it hardly affects the local structure and over-smoothing artifact is much more desirable.
- For data set with GT depth, edge are chosen with disparity map exceeding a threshold. Otherwise we use segmentation mask, and group FG and BG classes and then dilate the boundary with 3x3 kernel.

#### Notes
- Need to read papers on disparity estimation, including
	- [GCNet: End-to-End Learning of Geometry and Context for Deep Stereo Regression](https://arxiv.org/abs/1703.04309)
	- [PSMNet: Pyramid Stereo Matching Network](https://arxiv.org/abs/1803.08669)
	- [Practical Deep Stereo (PDS): Toward applications-friendly deep stereo matching](https://arxiv.org/abs/1806.01677) 
- In reality, over smoothing happens in many places, and typically it indicates a bimodal or multi-modal problem. It may make sense to reformulate the regression problem a classification based one.
