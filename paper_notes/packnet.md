# [PackNet: 3D Packing for Self-Supervised Monocular Depth Estimation](https://arxiv.org/abs/1905.02693)

_May 2020_

tl;dr: Improvement of neural net architecture to improve self-superv0sed depth. It leverage velocity when possible to achieve scale awareness.

#### Overall impression
The paper inherits all previous loss functions in self-supervised monocular depth method, and can be seen as a bag-of-trick for self-supervised mono depth.

The importance of high resolution has been demonstrated through [SuperDepth](https://arxiv.org/abs/1810.01849) <kbd>ICRA 2019</kbd> via super-resolution with channel2spatial.

Most previous method requires depth scaling at test time. When velocity is available for regularization (during training), the network outputs metric results at minor degradation of performance. **However it does not leverage this info during test time, and may fail at unseen ego motion.** See [SC-Sfmlearner](sc_sfm_learner.md) and [TrianFLow](trianflow.md) for other scale awareness solution.

#### Key ideas
- Use spatial2channel + 3D conv packing for more effective downsampling.
- The 3D packing + unpacking operation can preserve spatial details. Normal maxpool + bilinear upsample cannot keep original resolution. The code is relatively straightforward.
- The paper also adds a velocity loss when it is available.

#### Technical details
- DepthNet Regress inverse depth.
- The proposed PackNet also performs better at higher resolution and does not plateau as early as ResNet.
- MR (640x192) vs HR (1280 * 384). Note that the ground and sky are cropped out.

#### Notes
- [Github code](https://github.com/TRI-ML/packnet-sfm/)
- The authors mentioned that it is possible to replace the poseNet prediction with GT pose, but they did not see significant improvement. [github issue](https://github.com/TRI-ML/packnet-sfm/issues/39)

