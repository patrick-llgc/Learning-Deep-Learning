# [RoI Transformer: Learning RoI Transformer for Oriented Object Detection in Aerial Images](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_Learning_RoI_Transformer_for_Oriented_Object_Detection_in_Aerial_Images_CVPR_2019_paper.pdf)

_September 2019_

tl;dr: Learn parameters for rotated position sensitive pooling to eliminate the need for oriented anchors.

#### Overall impression
**Position sensitive (PS) roi pooling**: [R-FCN](rfcn.md) (PS RoI Pooling) --> [light head R-CNN](lighthead_rcnn.md) (PS RoI Align) --> [RoI Transformer](roi_transformer.md) (Rotated PS RoI Align)

Same authors from [DOTA](dota.md) dataset. Detecting oriented objects is an extension of general horizontal object detection. Three related fields: remote sensing, text scene detection.

The RoI Learner is quite similar to a bbox refinement stage in Faster RCNN, just added an additional dimension of orientation. It replaces the need for many orientated anchors. Alternatively, if oriented anchors are used, then RPN yields RRoI, and RRoIs are used to do oriented PS RoIAlign to regress bbox offsets (both size, position and orientation).


#### Key ideas
- RoI Transformer still uses horizontal anchors. The horizontal RoIs (HRoIs) are then rotated to be RRoIs
- RoI transformer:
	- RoI learner: Use HRoI aligned features to learn a transformer to get a rotated RoI. The HRoIs are external rectangle of RRoIs.
	- RoI warping: rotated position sensitive RoIAlign to crop rotation invariant features
	- Once RRoIs are obtained, there seems to be an additional step of matching GT to the RRoIs. --> this need to be explored by inspecting [source code](https://github.com/dingjiansw101/RoITransformer_DOTA).
- The regression in RRoI learner is relative to the rotated RoI (although this stage the RRoI is horizontal anchor boxes). The final regression of orientation refinement is also related to the rotated RoI. 
- **Contextual RRoI**: RoI aligned features are enlarged (1.2x1.4) to include more context. This improves AP by several points on [DOTA](dota.md).

#### Technical details
- Rotated ROI increases the number of anchors significantly, leading to training complexity
- Rotated RoI align crops rotation-invariant features. 
- Regression target in oriented RoI coordinates. This helps ensure rotation invariance. 
- GT and anchor matching: HRoIs (anchors) should be matched to oriented bbox, but for simplicity the HRoIs are matched to the external horizontal bbox of oriented GT bboxes. The proposed algorithm can learn parameters of RRoI from HRoI aligned features. 
- The lighthead RNN can be deployed with FPN. This is done by applying large separable convolution on P2-P5. 

#### Notes
- Can we apply centerNet to DOTA dataset?
- [source code](https://github.com/dingjiansw101/RoITransformer_DOTA)