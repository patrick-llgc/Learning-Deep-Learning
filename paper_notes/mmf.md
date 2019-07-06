# [MMF: Multi-Task Multi-Sensor Fusion for 3D Object Detection](http://www.cs.toronto.edu/~byang/papers/mmf.pdf)

_June 2019_

tl;dr: Use auxiliary tasks (ground estimation and depth completion) and sensor fusion boost 3D object detection. MMF is fast running at 13 FPS. 

#### Overall impression
This paper is built on [ContFuse](contfuse.md) and two-stage sensor fusion methods such as [MV3D](mv3d.md) and [AVOD](avod.md). MMF and ContFuse is similar to AVOD that it uses fused feature for proposal generation. And MMF and ContFuse method is **anchor-free**. However MMF is better than ContFuse in that it uses depth estimation for a dense pseudo-lidar point cloud.

The paper is also influenced by [HDNet](http://proceedings.mlr.press/v87/yang18b/yang18b.pdf) which exploits HDmap and estimates ground height for 3D detection. 

MV3D --> AVOD --> ContFuse --> MMF

This boost the 2D hard by more than 8 AP (from 80 to 88) among real-time models. (RRC from sensetime performs really well for 2D OD but runs at 3.6 s/frame)

#### Key ideas
- Cascaded design (such as F-Pointnet) suffers from the drawback from each stage. It has difficulty recovering object that are far away.
- Sensor fusion architecture: point-wise and ROI-wise feature fusion.
	- pointwise sensor fusion: use lidar points to establish correspondence between BEV and image (like in [contFuse](contfuse.md)). Multi-scale features from FPN of image is fused into each layer of lidar feat maps. 
	- ROI-wise sensor fusion: project 3D bbox to BEV and image space (with RoIAlign and oriented RoIAlign)
- Proposal generation: 1x1 conv on top of last BEV feat map (like YOLO). This feat map has feat from both image and BEV.
- Map fusion: estimated ground height is subtracted from the regression target. This eases the 3D object localization (don't we assume this from the beginning?)
- The image backbone is small (Resnet 18) but it benefits quite a bit from lidar sensor fusion.

#### Technical details
- The depth completion idea is very similar to [pseudo-lidar](pseudo_lidar.md). A depth map GT is used to supervise this task.
- By applying ground estimation and depth completion during training, the network learns a better representation, but no need to deploy during inference.
- Pretraining: resnet-18 and pretrained depth estimation

#### Notes
- 3D perception matters more than 2D as motion planning happens in bird's eye view.
- Honestly speaking, without the bbox finetuning, the performance is already pretty good for industry applications. This way we could harvest the benefit of Sensor fusion more easily.
- Q: why it is so fast? --> ResNet18 as backbone
- Q: how to do point wise feature fusion without accurate calibration?