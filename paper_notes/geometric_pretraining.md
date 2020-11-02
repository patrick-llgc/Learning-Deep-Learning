# [Geometric Pretraining for Monocular Depth Estimation](http://lewissoft.com/pdf/ICRA2020/0035.pdf)

_September 2020_

tl;dr: Use self-supervised optical flow loss to pretrain of a structure encoder for monocular depth estimator on uncalibrated videos. 

#### Overall impression
The paper provides a new direction to improve monocular depth estimation. Conventional ImageNet pretraining helps more with classification tasks than location-aware tasks such as object detection and depth estimation, as spatial information got discarded.

The core of the algorithm is still photometric loss, and has inherent limitations. 

One main difference between optical flow and depth estimation task is that optical flow estimation does not care about calibration.

The idea of encoding motion between two different frames into a latent vector is very similar to PoseNet in [SfM-learner](sfm_learner.md) and [Struct2Depth](struct2depth.md) and more flexible. 

The idea of using geometric pretraining to improve monoDepth or mono3D is similar to [CubifAE-3D](cubifae_3d.md).

#### Key ideas
- Pretraining Architecture
	- A conditional encoder-decoder to separate **structure** information and **motion** information
	- The PoseNet takes in two consecutive images and output a **128-dim latent motion vector**. --> This is similar to [GeoNet](geonet.md), but instead of regressing every single variable in the pose vector, it encodes the motion (ego motion and other motions of the car) into a latent vector. This is supposed to be more versatile.
	- The flow decoder learns the flow, and uses the self-supervised flow loss.
	- Only the structure encoder is kept as the backbone. It learns general structures and does not explicitly estimate depth and pose. 
- Then the backbone (structure encoder) is finetuned for depth estimation.

#### Technical details
- The reason to use optical flow as the pretraining task as optical flow only cares about pixel correspondence and does not care about extrinsics/intrinsics and thus we can use uncalibrated videos for this pretraining.

#### Notes
- [Video on Youtube](https://www.youtube.com/watch?v=wlobHEpPXDU) and [part2](https://www.youtube.com/watch?v=5AKuyMUk4w8)
- Q: is it possible to use single image for depth pretraining? The idea would be similar to [Virtual Cam/Movi-3D](movi_3d.md) and [Cam Conv](cam_conv.md), by augmenting single image and its depth value. 
