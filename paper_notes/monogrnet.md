# [MonoGRNet: A Geometric Reasoning Network for Monocular 3D Object Localization](https://arxiv.org/pdf/1811.10247.pdf)

_August 2019_

tl;dr: Use the same network to estimate instance depth, 2D and 3D bbox.

#### Overall impression
The authors raises a critical issue in the current depth estimation that the evaluation metrics are not focused on instance level depth estimation. That means all SOTA methods are sub-optimal in terms of estimating instance level depth. This point echoes mine that the pixel wise depth map need finetune for 3D object detection, as opposed to freezing it as done in [pseudo-lidar end2end](pseudo_lidar_e2e.md).

The depth estimation is instance based sparse depth info. --> There should be a way to marry depth estimation and 3D object detection together. Or is this sparse depth info is already enough? This is the first tunable depth estimation in almost all mono3DOD research. Most just use pretrained depth estimation off the shelf.

However the ablation study is a bit insufficient. It did not show what are the main reasons for missed detection. Is it due to the sub-par 2D object detector? 

#### Key ideas
- **Pixel level depth estimation does not optimize object localization by design.** --> We need to finetune the monocular depth estimator for 3D object detection (c.f. [pseudo-lidar end2end](pseudo_lidar_e2e.md)).
- Monocular 3DOD: predict 2D bbox, instance depth Zc, c (2D projection of C) and local corners Oi.
- The projection of the amodal 3d bbox is different from the 2D bbox center b. In particular, when object is cropped, b is always inside image but c may be outside FoV.
- Architecture:
	- 2d bbox branch is based on Uber's kittiNet
	- RoI aligned features are used to refine depth and projection of 3d bbox center, and regress the 8 corners in the local system (in the frame determined by observation angle).
- The training of the refinement are based on two losses, one based on the coarse estimation and GT, one based on the coarse estimation + refinement and GT). During inference, the final result is coarse estimation + refinement --> Also, I am not quite sure if the way the refinement are regressed are optimal.
- Directly regressing angles in the camera coordinates leads to worse results. This confirms that the local appearance can only infer observation angle, as [deep3dbox](deep3dbox.md) pointed out.

#### Technical details
- Depth estimation is though a depth encoder structure like DORN (CVPR 2018).
- Multi-stage training: 2dod is trained first, then 3dod, then finetune altogether.


#### Notes
- Q: why pick regressing 8 corners? The regressed 8 corners (16 DoF) may not form a 3D bbox anyway. Maybe regressing 7 DoF 3D bbox is a better idea (c.f., [frustum pointnet](frustum_pointnet.md) and [psedudo-lidar end2end](pseudo_lidar_e2e.md)).
