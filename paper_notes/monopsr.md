# [MonoPSR: Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction](https://arxiv.org/abs/1904.01690)

_July 2019_

tl;dr: 3DOD by generating 3D proposal first and then reconstructing local point cloud of dynamic object. 

#### Overall impression
This is from the authors of AVOD. The structure of the The centroid proposal stage is quite accurate (average absolute error ~1.5 m).

This paper is heavily influenced by [deep3dbox](deep3dbox.md), in particular the leverage of 2bbox and estimation of orientation and dimension as a first step.

The reconstruction branch regresses a local point cloud of the object and compares with the GT in point cloud and camera (after projection). The paper did not talk about the incremental boost of this branch, and seems to be just a fancy regularization branch (multi-task).

#### Key ideas
- **Shared feature map** is created by concatenating features ROIPooled from feature map, and features learned from image patches. (c.f. the two ROI pooling branches, one with tight bbox and the other with dilated bbox in [mono3d](mono3d.md)).
- Proposal **depth** initialized from the bbox height and real height of objects. This leads to more accurate depth estimation. It is even possible to regress the residual distance directly from the patch.
- Proposal stage
	- estimate the physical dimension (offset from class average) and orientation (mainly the local angle, or observation angle, NOT the yaw)
	- Estimate depth (from CV, bbox and object height stats)
- **The proposal stage is highly practical for 3D object detection**. Together with the refinement module the performance can be boosted even further. First depth z is calculated from the intrinsics and estimated bbox and the regressed physical height of the object (from a local patch). Then the x and y can be estimated by reprojecting the (u, v) center of the 2d bbox to the plane at depth z. 

#### Technical details
- Deep3DBox and other methods that use 2D bbox as prior has the disadvantage of locking in the error in 2D object detection.

#### Notes
- Did the paper regress yaw or the observation angle? The two are quite different as per [deep3DBox](deep3dbox.md). --> Not yaw, but local observation angle. (confirmed by first author)
- Need to be compared with other mono 3dod algorithms such as MLF and deep3DBox.

