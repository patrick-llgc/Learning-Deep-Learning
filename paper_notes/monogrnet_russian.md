# [MonoGRNet 2: Monocular 3D Object Detection via Geometric Reasoning on Keypoints](https://arxiv.org/abs/1905.05618)

_October 2019_

tl;dr: Regress keypoints in 2D images and use 3D CAD model to infer depth. 

#### Overall impression
The training is based on 3D CAD model with minimal keypoint annotation. This is valuable as it saves much annotation effort on 2D images, which is inefficient and inaccurate. It also seems to use the semi-automatic way to annotate 2D keypoints as in [deep MANTA](deep_manta.md).

It is related to [deepMANTA](deep_manta.md) that it relies on keypoint regression for monocular 3DOD. The idea of using keypoint to estimate depth can also be found in [GS3D](gs3d.md). It is not actually that related to MonoGRNet.

It follows the Mono3DOD tradition that regresses local yaw and dimension offset from image patches and infer depth from these results.

#### Key ideas
- The keypoint regression relies on multi-task reprojection consistency loss. This eliminates tedious keypoint annotation.
- 5 CAD model x 14 keypoints each.
- Architecture:
	- Mask RCNN object detection (not used, but essential to stablize training)
	- 2D keypoint regression sub-netwok: 2D keypoints, visibility and local orientation --> instead of heatmap, they directly regress keypoint coordinate through fc layers. 
	- CAD and dimension model: CAD cls + dimension offset. 
- Scale according to keypoint distance. This idea is valid as the dimension or keypoint distances do not vary much.
- Regress the local orientation, then convert to global orientation.
- Reprojection consistency loss to make the results from multiple heads consistent. (e.g., 2D 3D tight constraints, 2D keypoint location)

#### Technical details
- Dense depth estimation may be redundant in context of 3D object detection. [MonoGRNet](monogrnet.md) only regresses instance level depth. This paper focuses on salient features (keypoints). 
- Single stage 2D object detection does not easily extend to related tasks such as instance segmentation and keypoint regression.
- Four pairs of keypoints on the wind shield corner is used for depth retrieval. All others are used as additional supervision.
- The orientation angle difference is about 10 degrees. 

#### Notes
- In industry we can afford labeling keypoints and visibility, but this paper assures that the idea of using keypoint for geometric reasoning is valid due to small variations of geometric properties given a vehicle subtype.
- The idea of using the prior size info of some keypoints or vehicle size can be found in early works of [this CVPR 2012 paper](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/wojek12.pdf). This is said to be similar to mobileEye's approach.
