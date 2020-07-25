# [GeoNet: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose](https://arxiv.org/abs/1803.02276)

_July 2020_

tl;dr: Use ResFlowNet and consistency check to improve monodepth.

#### Overall impression
GeoNet decouples the pixel movement to **rigid flow** and **object motion** adaptively. The movement of static parts in a video is solely caused by camera motion. The movement of dynamic object is caused by camera motion and specific object motion.

However it is still using photometric error instead of geometric error (unlike in [DeepV2D](deepv2d.md)).

#### Key ideas
- PoseNet and DepthNet similar to [SfM Learner](sfm_learner.md).
	- We will have naturally a rigid flow from depth and pose estimation. 
- The main idea is to use ResFlowNet to predict the **residual** flow to handle non-rigid cars. 
	- As compared to predict global flow, predicting residual flow is easier
	- **Caution**: ResFlowNet is extremely good at rectifying small errors from rigid flow, but cannot predict large optical flow, and need to be addressed with additional supervision. Regressing global flow does not have this issue.
- A geometric consistency check is performed to mask inconsistent loss. It is similar to that used in optical flow or left-right consistency loss in [monodepth](monodepth.md).
	- If warped target pixel and source pixel difference is smaller than 3 pixel or 5% error, then use the loss at that location. Otherwise mask it out.

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

