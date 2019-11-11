# [MVRA: Multi-View Reprojection Architecture for Orientation Estimation](http://openaccess.thecvf.com/content_ICCVW_2019/papers/ADW/Choi_Multi-View_Reprojection_Architecture_for_Orientation_Estimation_ICCVW_2019_paper.pdf)

_November 2019_

tl;dr: Build the 2D/3D constraints optimization into neural network and use iterative method to refine cropped cases.

#### Overall impression
This paper is heavily based on [deep3Dbox](deep3dbox.md) and adds a few improvement to handle corner cases.

The paper has a very good introduction to mono 3DOD methods. 

#### Key ideas
- 3D reconstruction layer: instead of solving an over-constrained equation, MVRA used a reconstruction layer to lift 2D to 3D. 
	- **IoU loss** in perspective view, between the reprojected 3D bbox and the 2d bbox in IoU. 
	- L2 loss in BEV loss between estimated distance and gt distance.
- **Iterative orientation refinement for truncated bbox**: use only **3 constraints instead of 4**, excluding the xmin (for left truncated) or xmax (for right truncated) cars. Try pi/8 interval and find best, then try pi/32 interval to find best. After two iterations, the performance is good enough.

#### Technical details
- Bbox jitter to make the 3D reconstruction layer more robust.

#### Notes
- The use of IoU to pick the best configuration is proposed before in [Shift RCNN](shift_rcnn.md).
- The BEV loss term can be used to incorporate radar into training process. 
