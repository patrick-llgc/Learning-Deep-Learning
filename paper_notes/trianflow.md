# [TrianFlow: Towards Better Generalization: Joint Depth-Pose Learning without PoseNet](https://arxiv.org/abs/2004.01314)

_May 2020_

tl;dr: Use optical flow and dense 2D-2D to solve for local pose and align with depth prediction. 

#### Overall impression
The name seems to come from "triangulate-flow". 

PoseNet lack generalization ability (performs badly for long sequence where relative pose across sequence is hugely different, when video is speed up, and also [hardly beats image retrieval baseline](understanding_apr.md)).

The idea of using optical flow to calculate relative pose is very similar to [DF-VO](df_vo.md). The main difference 

- [DF-VO](df_vo.md) has pre-trained depth and flow network separately with PoseNet-like architecture while [TrianFLow](trianflow.md) got rid of PoseNet altogether and uses the triangulation to perform self-supervision.
- [DF-VO](df_vo.md) is based on [SC-sfm-learner](sc_sfm_learner.md) to ensure consistency and aligns pose to depth. [TrianFlow](trianflow.md) aligns depth to pose. (Why?)

The knowledge of correspondence (matching) does not have to be learned by PoseNet and thus improves network generalization ability.

#### Key ideas                                                                                                                                                                                                                                                  
- FlowNet is based on [PWC-Net](pwcnet.md)
- Scale is explicitly disentangled at both training and inference.
- Training: 
	- optical flow to get dense matching
	- forward-backward consistency to generate score map Ms
	- Sample points that survives occlusion mask Mo and top 20% forward-backward score.
	- 8 pt algorithm in RANSAC + cheirality check to solve F matrix and R|t.
	- Based on R|t and correspondence, get triangulated point depth with **mid-point triangulation** to get up-to-scale 3d structure. Points around epipoles (vanishing points) are removed for triangulation.
	- Dense predicted depth is **aligned** to sparse triangulated depth. The 3d structure's scale is determined by relative pose scale. The triangulated depth is used as pseudo-depth signal to supervise depth prediction
- Inference (same as [DF-VO](df_vo.md))
	- Calculate fundamental matrix from optical flow
	- When optical flow is too small, use PnP to solve for relative pose. 
- TrianFlow can generalize to unseen ego motion. 
	- For 3x fast sequence, ORB-SLAM2 frequently fails and reinitializes under fast motion
- The results is better than most other end-to-end methods, but not a good as [DF-VO](df_vo.md).

#### Technical details
- Occlusion map, Mo
- Flow consistency score map, Ms
- inlier score map, Mr, by computing distance map from each pixel to its corresponding epipolar line. [Implementation of inlier mask in code](https://github.com/B1ueber2y/TrianFlow/blob/f8b3e77d172b61b5fb395801f42d2d83e61e3d0d/core/networks/model_triangulate_pose.py#L54)
- Angle mask: filter out points close to epipoles. [Implementation of angle mask](https://github.com/B1ueber2y/TrianFlow/blob/f8b3e77d172b61b5fb395801f42d2d83e61e3d0d/core/networks/model_depth_pose.py#L147)

#### Notes
- Q: the scale normalization is there to ensure a consistent scale between depth and flow, but what ensures a scale consistency across frames? --> This is done in a similar manner to vSLAM?
- The paper 
- During inference, the code actually assumes [depth predictions have consistent scale]() and thus [aligns pose to depth](https://github.com/B1ueber2y/TrianFlow/blob/master/infer_vo.py#L162).	

> The central idea of existing self-supervised depth-pose learning methods is to learn two separated networks on the estimation of monocular depth and relative pose by enforcing geometric constraints on image pairs. 