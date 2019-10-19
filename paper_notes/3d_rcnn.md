# [3D-RCNN: Instance-level 3D Object Reconstruction via Render-and-Compare](http://openaccess.thecvf.com/content_cvpr_2018/papers/Kundu_3D-RCNN_Instance-Level_3D_CVPR_2018_paper.pdf)

_October 2019_

tl;dr: Mono 3DOD by estimating pose and shape of vehicles and render-and-compare loss.

#### Overall impression
3DOD is critical for prediction and path planning. However 3D ground truth is hard to obtain. 3D RCNN only needs 2D annotation (depth and semantic segmeantion). It also need accurate intrinsics/extrinsics to make it work.

This video seems to stem from the concept of this video of [PASCAL 3D](https://www.youtube.com/watch?v=5Yeus0x6fo0&feature=youtu.be)

First learn the low-dimensional space from CAD models for each subtype. PCA is used. But AutoEncoder seems also OK, such as [RoI10D](roi10d.md) which are heavily inspired by this work and seems more practical.

*Analysis by synthesis*: Estimate the shape, pose and size parameters of the cars, and render (synthesize) the scene. Then the mask and depth map are compared with ground truth to generate loss.

The shape and pose are weakly supervised and arise from end-to-end training.

#### Key ideas
- Estimate 2D bbox, 3D center projected on 2D from RoIAligned features. Estimate shape and pose with RoIAligned feature concatenated with the intrinsics of virtual RoI camera.

#### Technical details
- 10 dim shape space, slightly different from the 6d space by [RoI10D](roi10d.md).
- The authors argue that it is hard to predict 3D property such as shape and pose from RoIAligned features. --> [RoI10D](roi10d.md) did use the RoIAligned features.
- Improved multi-bin loss: weighted sum of bin center by confidence score, and L1 loss. This way there is no need to regress for the residual like the [deep3dbox](deep3dbox.md). 
- Render and compare uses operation of CUDA-OPENGL, and seems quite engineering heavy to make this work.

#### Notes
- 3D dataset with pose estimation
	- [Pascal 3D](http://cvgl.stanford.edu/papers/xiang_wacv14.pdf)
	- [multiview car dataset](https://www.epfl.ch/labs/cvlab/data/data-pose-index-php/): cars on turntable
- PCA is simple and efficient, but [RoI10D](roi10d.md) reported degeneracy of PCA models and favors 3D auto-encoder.