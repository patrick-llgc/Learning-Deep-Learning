# [DeepV2D: Video to Depth with Differentiable Structure from Motion](https://arxiv.org/abs/1812.04605)

_July 2020_

tl;dr: Video to depth with iterative motion and depth estimation.

#### Overall impression
The structure of the system seems to be a bit convoluted, where the training and inference are quite different.

Self-supervised methods such as [SfM-learner](sfm_learner.md) and [Monodepth](monodepth.md) uses geometric principles for **training** alone, and doe no use multiple frames to predict depth at inference. Similar works include **DeMoN and DeepTAM and MVSNet**.

This work also proposes to use geometric constraint, but instead of minimizing photometric error in [LS-Net]() or feature-metric error in [BA-Net](banet.md)), Flow-SE3 module mimimizes geometric reprojection error (difference in pixel location) leads to better well-behaved optimization problem. --> this idea is explored in [GeoNet](geonet.md) which jointly optimizes optical flow and depth and motion.

[DeepV2D](deepv2d.md) is similar to [BA-Net](banet.md).

- [BA-Net](banet.md) tries to optimize one joint nonlinear optimization over all variables, and thus needs to decompose depth prediction with depth basis to reduce search space.
- [DeepV2D](deepv2d.md) decomposes joint optimization into more tractable subproblems of optimization of depth and motion, and do block coordinate descent. It allows the depth estimation module to be more expressive and thus directly estimate per-pixel depth.
- Performance of [DeepV2D](deepv2d.md) is better than [BA-Net](banet.md) across the board. 

It seems to be more practical than [Consistent Video Depth Estimation](consistent_video_depth.md) as it converges quickly during inference (5-6 iterations).


#### Key ideas
- Two modules: motion estimation and depth estimation
- Motion estimation
	- PoseNet to predict initial pose. Both DeMoN and DeepTAM used this alone.
	- Flow-SE3. Takes depth as input and estimate dense 2D correspondence between frame pairs (optical flow?). Unroll one iteration of Gauss-Newton update over SE3 perturbation to minimize **geometric** reproj error. 
	- The error then goes through a LS-optimization layer and generates perturbation $\xi$. The perturbation updates G + $\xi$ --> G'. Then G' is supervised by geometric error of points with predicted depth projected onto the target image.
	- Pose optimization:
		- global pose optimization: all frames need to have depth. In inference this mode can be used. 
		- keyframe pose optimization: only one frame need to have depth. This is used for training due to memory efficiency.
		- Verdict: Not too much difference in performance. It can jointly optimize pose over multiple frames rather than PoseNet which is able to optimize pairwise motion.
		- ![](https://user-images.githubusercontent.com/11929093/85421801-d852de80-b542-11ea-939c-f992950a04e2.png)
- Depth estimation
	- Use multiview stereo to reconstruct cost volume (similar to [GCNet](gcnet.md) for stereo matching). DL is used for both feature extraction and matching.
	- Uses depth GT for supervision --> can we do without this?

#### Technical details
- When the 3D stereo network is replaced by a 2D encoder-decoder network the depth accuracy is significantly worse.
- DeepV2D is trained with fixed 4 frames as input.
- DeepV2D quickly converges with a small number of iterations (during inference), and improves with more frames. 

#### Notes
- [Code on github](https://github.com/princeton-vl/DeepV2D)
- Feature Matching vs feature tracking

