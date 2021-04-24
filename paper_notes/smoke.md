# [SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation](https://arxiv.org/abs/2002.10111)

_June 2020_

tl;dr: Mono3D based on [CenterNet](centernet.md) and [monoDIS](monodis.md).

#### Overall impression
The paper is a solid engineering paper as an extension to [CenterNet](centernet.md), similar to [MonoPair](monopair.md). It does not have a lot of new tricks. It is similar to the popular solutions to the [Kaggle mono3D competition](https://www.kaggle.com/c/pku-autonomous-driving).

A quick summary of [CenterNet](centernet.md) monocular 3D object detection.

- [CenterNet](centernet.md) predicts 2D bbox center and uses it as 3D bbox center. 
- [SMOKE](smoke.md) predicts projected 3D bbox center.
- [KM3D-Net](km3d_net.md) and [Center3D](center3d.md) predict 2D bbox center and offset from projected 3D bbox center. 
- [MonoDLE](monodle.md) predicts projected 3D bbox center, and also predicts offset to 2D bbox as auxiliary task.

#### Key ideas
- SMOKE eliminates 2D object detection altogether. Instead of predicting the 2d bbox center and the 3d/2d center offset, SMOKE predicts 3D center directly. --> This may have some issues as for cars heavily truncated, the 3D center may not be inside the image. 
- Rather than regressing the 7 DoF variables with separate loss functions, SMOKE transform the variables into 8 corner representation of 3D boxes and regress them with **a unified loss functions**. This is a nice way to implicitly weigh the loss functions. (cf [To learn or not to learn](to_learn_or_not.md) which regresses an essential matrix.)
- **Disentangles loss** from [monoDIS](monodis.md) groups the 8 parameters into 3 groups. In each group, use the prediction in that group and the gt from other groups to lift to 3D and calculate overall loss. The final loss is an unweighted averaged of the loss from different group. 
- Classification
	- Projection of 3D center is predicted as a virtual keypoint via heatmap, similar to that in [CenterNet](centernet.md).
- Regression
	- Regresses 8 parameters for 7 DoF (cos and sin for angle). Normalize the regression target to ease training. The prediction bit is after sigmoid. 


#### Technical details
- Data augmentation only used to regress keypoint. 
- When a car's 3D center is outside the image, discard it. This is about 5% of all objects. 
- Runs real-time at 30 ms per frame with Titan XP.
- The distance estimation is quite good. About 3 meter at 60 meters. Less than **5% error**. This is much better than [frontal obj distance estimation](obj_dist_iccv2019.md) by NYU and xmotors.ai.
- 3D --> 2D bbox also achieves very good results than many 2D --> 3D method. This shows 3D object detection can have more robust detection results. 
- Ablation Details
	- [GroupNom](groupnorm.md) > BN
	- Dis L1 > L1 > smooth L1. 
	- Vector (sin, cos) > Quaternion representation

#### Notes
- [Code on github](https://github.com/lzccccc/SMOKE)
- Need to implement the 2D center prediction and offset between 2D and 3D to recover heavily truncated 3D bbox. This method can be extended to other scenarios where the predicted location goes out of a ROI. See [KM3D-Net](km3d_net.md) and [Center3D](center3d.md).

