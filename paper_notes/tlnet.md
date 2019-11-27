# [TLNet: Triangulation Learning Network: from Monocular to Stereo 3D Object Detection](https://arxiv.org/abs/1906.01193)

_October 2019_

tl;dr: Place 3D anchors inside the frustum subtended by 2D object detection as the mono baseline. The stereo branches reweigh feature maps based on their coherence score.

#### Overall impression
Pixel level depth maps are too expensive for 3DOD. Object level depth should be good enough. --> this is similar to [MonoGRNet](monogrnet.md).

The paper provides a solid mono baseline. --> this can be perhaps improved by using some huristics such as vehicle size to overcome the dense sampling of 3D anchors.

The paper still requires the 3D bbox GT and stereo instrinsics for the training of the monocular detection network. --> Maybe annotate directly on 2D images?

#### Key ideas
- Dense placement of 3D anchors (0.25 m interval) in [0, 70m] ranges, with two orientations (0 and 90 deg) for each object class, with the avg size of the object class.
- Triangulation: if one proposal tightly bound the GT, then the left and right projected ROI should tightly bound the GT projection as well. The RoIAligned features from left and right images are fused to regress for the overall confidence.

#### Technical details
- RPN and refinement stage: The RoIAligned features are used to regress the size and location offset. Then the adjusted region proposals are RoIAligned again to regress size and location offset, and orientation offset. 
- **Coherence score weighting**: even without explicit supervision, the roiAligned features focuses on some of the keypoints. This can be further reinforced by reweighing the channels with channel-wise cosine similarity (coherence score) between left and right feature maps. --> This is similar to SENet, but instead of self-attention, this coherence weighting has physical significance.
- Coherence weighting > summation > concatenation

#### Notes
- Questions and notes on how to improve/revise the current work  

