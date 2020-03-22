# [Monocular 3D Object Detection with Decoupled Structured Polygon Estimation and Height-Guided Depth Estimation](https://arxiv.org/abs/2002.01619)

_March 2020_

tl;dr: Decouple 2D projection estimation with depth estimation.

#### Overall impression
The paper is the first one to state clearly that the idea that the depth dimension is decoupled from predicting the 2D projection of 3D cuboid. Essentially it says we can estimate the 3d bbox position at normalized iamge plane, then estimate the depth. 

This is the correct way to formulate the question and I am surprised that no previous work has formulated this question this way. [MonoDIS](monodis.md) tried to disentangle the losses but it is a more general framework from the training's perspective.

#### Key ideas
- Treat all eight points of a cuboid as keypoints and predict through a stacked hourglass keypoint detector. This is similar to [RTM3D](rtm3d.md).
- Use the four apparent height of the vertical edges of the cuboid (structured polygon) to estimate depth of the 4 vertical edges (and thus 8 vertices). The height for the object is actually regressed from the RoI pooled features $t_H = \log(G_H/A_H)$, A_H = 1.46 m is the average height. Network predicts $t_H$ but recovers $G_H$ for depth estimation.
	- **Over-parameterization lead to postprocessing during inference**: Once we have the 3D cuboid, calculate h, w, l with the average of four edges, and orientation is average of the orientation of four edges. 

#### Technical details
- X (lateral) from -25 to +25, Z (frontal) from 0 to 50 m. 
- Height guided distance estimation reduces the std of depth from 2m to 1m. 

#### Notes
- From fig. 8 it looks  they did not show how much yaw improved before and after the lidar finetune

