# [AFDet: Anchor Free One Stage 3D Object Detection](https://arxiv.org/abs/2006.12671)

_July 2020_

tl;dr: [CenterPoint](centerpoint.md) with bells and whistles wins the 2020 Waymo Open Dataset challenge.

#### Overall impression
The anchor-free object detection of [AFDet](afdet.md) is very close to [CenterPoint](centerpoint.md).

#### Key ideas
- Anchor free, NMS free. 1000x faster on CPU. Embedded system friendly.
- Architecture
	- [PointPillars](pointpillars.md) backbone.
	- The backbone only has two stage network and keeps the **same high resolution** for object detection. No FPN needed as in BEV only one scale is needed.
	- Same feature map size as input size. Do not downsize.
- Lidar pseudo-image are more sparse than natural images. 
	- Larger kernel and more non-zero pixels than Gaussian. Fill in all pixels inside bbox to a small value. --> [CenterPoint](centerpoint.md) enlarged gaussian kernel as well.
	- More pixels contribute to offset regression.
- Offset branch not only corrects quantization error, but can also correct regression error. 5x5 pixels to regress offset, as compared to only 1 pixel regressing offset in CenterNet.
- Data augmentation:
	- Create a bank of objects
	- randomly select 15 GT samples for car/vehicle and place them into the current point cloud. 
	- Each object is rotated [-9, 9] deg
	- global rotation [-45, 45] deg
- Bag of tricks
	- High resolution input matters
	- High resolution feature map helps
	- AdamW + 1 cycle policy for super convergence.
	- Data aug during training
- Tricks for winning solution (not in this paper)
	- Densification (with pervious 4 frames)
	- pointpainting (2D bbox painting and segmentation painting). 50% painted as waymo does not have a rear view cam
	- train 3 models and perform ensemble and TTA
	- Merging of TTA and ensemble bbox with [weighted bbox fusion](https://arxiv.org/abs/1910.13302)


#### Technical details
- 8 bits for orientation (2 bins x (cos + sin + 2-bit cls) per bin) --> This is same as [CenterNet](centernet.md).
- Waymo and KITTI uses the same 0.7 3D IoU as the KPI.

#### Notes
- [Talk at CVPRW 2020](https://youtu.be/9g9GsI33ol8?t=974)

