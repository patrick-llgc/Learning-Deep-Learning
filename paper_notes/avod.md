# [Joint 3D Proposal Generation and Object Detection from View Aggregation](https://arxiv.org/pdf/1712.02294.pdf)

_Mar 2019_

tl;dr: AVOD is a sensor fusion framework that consumes lidar and RGB images. Use multimodal feature fusion for generating proposal, and multimodal features are ROI-pooled to perform classification and bbox regression.

#### Overall impression
This work is heavily influenced by [MV3D](mv3d.md). Both work are inspired by Faster RCNN, but AVOD improved MV3D in several ways. One area is the region proposal where MV3D uses only BV for proposal generation while AVOD uses both BV and RGB for region proposal. AVOD also improves MV3D by improving bbox encoding, heading angle regression and using FPN to improve detection of small objects.

#### Key ideas
- AVOD uses Proposes non-oriented and axis-aligned region proposals. The dimensions of anchors are determined by clusteirng the trianing samples.
	- The center (tx, ty) are sampled every 0.5 m, and tz are fixed at sensor height above ground.
	- The RPN regresses (dtx, dty, dtz, ddx, ddy, ddz).
- FPN is used to generate high resolution feature maps for small object detection
	- The study examined the different 3D recall performance on cars, cyclists and pedestrians and found out the cyc and ped have lower recall (small objects).
- The network uses 1x1 conv to reduce the dimension of the global feature map to reduce inference memory footprint (down to 2GB).
- Second stage bbox encoding: coordinates from four points of the bottom surface, and two heights (top surface and bottom surface), in total 10 numbers. This is greatly reduced from the 3x8=24 points loss in MV3D. (But this is still redudant as DoF is 7).
- There is an explicit angle refinement stage that determines the heading angle from the four possible directions. This was found to be more effective than regress the angle directly.
- **Ablation on 3D recall**: Adding RGB image as input does not improve the recall for cars, but improved ped and cyc a lot. Using FPN also helps greatly.
- The final results is improved upon MV3D, but the Frustum RCNN seems to dominate small objects such as ped or cyc.

#### Technical details
- 3D input data has lower resolution than RGB, and deteriorates as a function of distance.
- Note that there are two steps that uses ROI pooling. RPN uses it to determine if an anchor box is FG/BG and refines bbox. The second stage uses the refined bbox to pool global feature maps for further refinement and classification.
- KITTI validation results are used more often in papers as there are limitation on the submission to the test set.
- The 3D IoU does not tell the whole story of how an oriented bbox matches the GT. Even if the IoU is 1, the orientation may be opposite. This is measured by Average Orientation Similarity.
 
#### Notes
- How to remove empty anchors (by integral images)?

