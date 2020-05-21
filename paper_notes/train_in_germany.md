# [Train in Germany, Test in The USA: Making 3D Object Detectors Generalize](https://arxiv.org/abs/2005.08139)

_May 2020_

tl;dr: The main obstacle to generalized 3D object detection is the size stats of 3D objects in different locations.

#### Overall impression
This is another insightful work from the same authors of [Pseudo-lidar](pseudi_lidar.md) and [Pseudo_lidar++](pseudo_lidar++.md). The paper founds out that the size stats affect the localization performance (IoU=0.7) of 3D object detector. However detection at lower IoU thresholds does not degrade much.

The paper converted recently released lidar datasets (KITTI, nuScenes, Argoverse, Lyft, Waymo) into the format of KITTI, including car/truck taxonomy, truncation/occlusion level, difficulty levels. This is quite engineering heavy work.

#### Key ideas
- A detector trained on KITTI performs 36 percent worse on Waymo compared to the one trained on Waymo. With proposed correction, this gap can be largely closed to within a few percent.
- At IoU 0.7, there is a large domain gap. However if IoU threshold is relaxed, the gap is not that huge any more. So the **detection** performance does not degrade much, but the more fine-grained **localization** performance drops.
- Object detectors trained on KITTI is the worst to transfer to other datasets. KITTI dataset may be too limited.
- Correction methods:
	- FS: Few shot finetune. Finetuned with **20 labeled scenes** is nearly the same as the performance as trained from 500 images in the target domain from scratch.
	- SN: statistical normalization. Grab the points in the car, reshape it by average size delta, and paste back to the scene.
	- OT: output transformation. This seems to overcorrect and may lead to degradation.

#### Technical details
- KITTI difficulty is based on bbox sizes. This is not a good metric to compare across multiple datasets due to sensor difference. Therefore this paper proposes to use distance metric. Occlusion metric does not depends on image size and sensor focal length and can be used without change.
- The paper proposed a systematic way to calculate occlusion level and truncation level
	- Truncation: bbox size within Fov/amodal bbox size
	- Occlusion: pixels that is not in other bboxes/bbox size.
- Datasets:
	- KITTI: 64-liner
	- nuScenes: 32-liner
	- Argoverse: two 32-liner stacked vertically
	- Lyft: 40-liner roof + 2 x 40 bumper lidar
	- Waymo: one mid-range lidar on roof (75 m)  + 4 side short-range lidar (20 m)
- Lyft is the easiest by AP score. NuScenes is hardest.
- Nuscenes lidar points per car is almost 1/10 of that of KITTI and Waymo. Lidar points per scene as well.

#### Notes
- [Github code](https://github.com/cxy1997/3D_adapt_auto_driving)

