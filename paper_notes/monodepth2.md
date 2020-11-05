# [Monodepth2: Digging Into Self-Supervised Monocular Depth Estimation](https://arxiv.org/abs/1806.01260)

_June 2019_

tl;dr: Three tricks to handle occlusion/dis-occlusion/relatively static objects/static scenes better.

#### Overall impression
Monodepth is on estimating depth using stereo pairs. This paper digs into videos. It is heavily influenced by the CVPR2017 paper [sfm learner](sfm_learner.md). The changes to losses are surprisingly simple, yet effective. In particular, the selecting of the min of many losses is very similar to the idea in [FSAF](fsaf_detection.md). 

This paper has a very solid code base which has become the foundation of many future works (such as [D3VO](d3vo.md)).

#### Key ideas
- Per pixel min reprojection loss: at each pixel, instead of averaging the reprojection loss, use the min of loss in all the images. This improves the sharpness of occlusion boundaries.
- Auto-masking stationary pixels. This filters out pixels which do not change appearance from frame to the next. This per pixel mask is calculated in forwarding pass, instead of learned such as in [sfm learner](sfm_learner.md).
	- This criterion indicates a static camera, or static object (relative to ego), or a low texture region.
	- This helps solving some of the inf depth issue.
- Scale back to original scale then do photometric loss calculation. This helps removing holes in large low-texture region. 

#### Technical details
- Edge preserving loss with edge-aware L1 loss, as in [monodepth](monodepth.md) (2017 CVPR oral).

	> We encourage disparities to be
locally smooth with an L1 penalty on the disparity gradients ∂d. As depth discontinuities often occur at image gradients, similar to [21], we weight this cost with an edge-aware term using the image gradients ∂I,

#### Notes
- [Codes](https://github.com/nianticlabs/monodepth2)
- Using consistency as supervision can be considered to be unsupervised (from the perspective that it does not require manual labels), or self-supervised. This is similar to VAEs.
- Q:  I would expect there are still a lot of pixels in the relatively static object that does not satisfy this requirement and thus are used in the loss. Maybe adding a scaling factor, say, 1.2 (another hyper-parameter) to the RHS of the inequality of Eq. 5 is better in that we are more stringent/conservative in selecting which pixels contribute to the loss. Once we set up the environment we could give it a quick try