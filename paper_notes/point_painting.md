# [PointPainting: Sequential Fusion for 3D Object Detection](https://arxiv.org/pdf/1911.10150.pdf)

_December 2020_

tl;dr: Augment point cloud with semantic segmentation results

#### Overall impression
The paper proposes a general method to fuse image results with lidar point cloud. This leads to minimum change to existing lidar detection networks and gives decent boosts to performance. 

Lidar and image are complementary: the point cloud provides accurate range view but with low resolution and texture info. The image has inherent depth ambiguity but offers fine-grained texture and color information. The addition of image info with lidar should boost lidar performance. 

The paper has a nice summary of previous camera-lidar early fusion methods and why they do not work well.

#### Key ideas
- Perform semantic segmentation, and append the class channels to lidar point cloud (KITTI C=4, and nuScenes, C=11)
- The representation of semantic results (one-hot or soft scores) does not matter
- **Pipelining** helps PointPainting to be integrated in a realtime system. The temporal delay up to 1 frame does not matter. Lidar can use the semantic segmentation results from the previous frame without degradation in performance.
- Higher semantic segmentation quality (with lidar points inside 3D bbox as oracle) can improver performance. 

#### Technical details
- [PointPillars](point_pillars.md) + [CBGS](cbgs.md) = PointPilars+, 10% higher mAP (30.5 --> 40.1)
	- Higher resolution: 0.25 m/pixel --> 0.2 m/pixel
	- More layers in the backbone
	- sample based weighting 
	- Global yaw augmentation pi --> pi/6.

#### Notes
- Questions and notes on how to improve/revise the current work  

