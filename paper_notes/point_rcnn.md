# [PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud](https://arxiv.org/pdf/1812.04244v1.pdf)

_Mar 2019_

tl;dr: Extend existing point cloud classification and segmentation method to 3D detection (instance segmentation and amodal bbox estimation). The PointRCNN directly generates high quality proposals from point cloud.

#### Overall impression
This paper basically extended Faster RCNN and Mask RCNN to point cloud representations, with many tweaks to adapt the representation difference. This work is closely related to other 3D detection framework such as [Frustum PointNet](frustum_pointnet.md), in particular the study in the appendix where proposals are generated from bird's eye view from point cloud. The main contribution is the first stage which generates high quality 3D bbox proposals from point cloud.

The idea of predicting one bbox per point is also used in [LaserNet](lasernet.md).

#### Key ideas
- Frustum pointnet uses 2D RGB image, but the hard examples it misses may be easy from 3D point cloud.
- In the first stage, the network performs semantic segmentation first (foreground vs background). This result is fed into the bbox proposal head (instead of independent as mentioned in the paper). Each point in the foreground is responsible of generating one bbox proposal (7 DoF). This eliminates expensive anchor based methods. The bboxes go through NMS before feeding to the 2nd stage.
- The point based proposal method generates higher recall than anchor based methods (such as AVOD). Anchor based methods inevitably assumes a bin size, leading to potential information loss and inefficient computation (same drawbacks as the voxelization methods).
- The second stage performs bbox refinement. It first transforms the points inside the proposal bbox (with a context margin $\eta$) using a canonical transformation. This transformation is critical.
	- The second stage takes in 512 points (with semantic features from stage 1, canonical coordinates, reflection insentity)
- Loss function:
	- Focal loss is used in the first stage (highly imbalanced classification problem).
	- **Full bin based regression** is very close to the cls-reg method in Frustum PointNet. 
	- The decision whether to use a bin-based regression (for x, z and $\theta$) or not (for y, h, w, l) really depend on whether the value follows a **unimodal distribution**.
	- Using **anchor box** is also an alternative to effective method for regression of non-unimodal distribution.

#### Technical details
- 3D objects are naturally and well separated by annotating 3D bounding boxes. ("the auto-labeled segmentation ground truth is in general acceptable" in Frustum PointNet.)
- The backbone is pointnet++ with MSG, but it can be others (pointnet or pointsift).
- Different from F-pointnet, which has to use a t-net to transform the pointnet to a canonical orientation, point rcnn performs a canonical transformation (translation and rotation) to point x to the heading direction of the bbox proposal.
- The training process also used synthetic data (pasting object in point cloud into the scenes). This improves convergence rate and 

#### Notes
- Q: Why generating point cloud from Birds View not enough? (doing an ablation study to verify this)
- Q: How much is the contribution from second stage? 
- Code not available yet
