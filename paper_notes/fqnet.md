# [FQNet: Deep Fitting Degree Scoring Network for Monocular 3D Object Detection](https://arxiv.org/abs/1904.12681) 

_September 2019_

tl;dr: Train a network to score the 3D IOU of a projected 3D wireframe with GT. 

#### Overall impression
This paper extends the idea of [deep3dbox](deep3dbox.md) beyond tight fitting, as deep3dbox depends much on the performance of 2D object detector. If the 2D object detector is inaccurate, then it will greatly affect 3D box accuracy.

The idea is to add a refinement stage to deep3dbox by densely sample around the 3D seed location (obtained by tight 2D/3D constraints), then score the 2D patches with rendered 3D wireframes. 

The idea is quite clever, but the optimization step to generate the 3D seed location is very time consuming and not very practical. 

The idea of [shift RCNN](shift_rcnn.md) and [FQNet](fqnet.md) are quite similar. Both builds on [deep3Dbox](deep3dbox.md) and refines the first guess. But [FQNet](fqnet.md) passively densely sample around the GT and train a regressor to tell the difference to GT, [shift RCNN](shift_rcnn.md) actively learns to regress the difference.

#### Key ideas
- Background: Patch based network to regress the local orientation and dimension. --> this is the same as [deep3dbox](deep3dbox.md) and [MonoPSR](monopsr.md), as vehicle dimension has a smaller range of variation and thus easier to regress. Note that MonoPSR regressed distance anyway. 
	- Dimension is closely related to the subtype of a car, which can be told by appearances. 
	- It is not quite practical to regress the distance from the patch itself as faraway and closeby objects have essentially the same appearances. --> but size of bbox can be of highly useful. 
- New contribution: FQNet. 
	- Intrinsics/extrinsics are necessary for projection
	- **Project 3D bbox to 2D image as wireframe.** Then pretrain it to classify if the patch contains the artificially painted bounding box or not, and then regress the IoU. 
- In the appendix, one important finding is that the azimuth angle scales with the center position of the bbox **linearly**. $\theta_{ray} = k (\frac{width}{2} - \frac{x1 + x2}{2})$. This should largely hold true for any lens without large optical distortion (non-fisheye lens).

#### Technical details
- The use of anchor-based regression breaks down the wide range of regression target into bins, and have a softmax cls loss + regression loss. This is similar to multi-bin loss.
	- the angle loss used **1-cos**, which is basically L2 norm.
	- Dimension loss is **3D IOU**, this actually dynamically weighs the contributions of 3 dimensions, to avoid cases where two dimensions matches but the third does not.


#### Notes
- Instead of generating the 3D seed location from time-consuming optimization. Can we do a coarser guess with IPM?
- Instead of "passively" generating dense samples, can we directly train a neural network to "actively" adjust the location of the 3D wireframe? --> see [shift rcnn](shift_rcnn.md). Yes, this work is also extended to [RAR-Net](rarnet.md) which trains a DRL agent to adjust the offset. 
- The active optimization can be done in a similar way to [IoU Net](iou_net.md) by gradient ascent via a differentiable criterion.
