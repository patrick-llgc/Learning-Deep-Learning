# [STSU: Structured Bird's-Eye-View Traffic Scene Understanding from Onboard Images](https://arxiv.org/abs/2110.01997)

_October 2021_

tl;dr: DETR-like structure for structured BEV perception of lane line and objects.

#### Overall impression
The paper focuses on the structured representation of the road networks and instance-wise identification of the traffic agents. This is a follow-up work to [BEV feature sticthing](bev_feat_stitching.md).

This paper follows the [DETR](detr.md)-style end to end object detection (extended to structured lane detection), which uses sparse queries in BEV space. This is actually one direction of Tesla's future work as they mentioned in AI Day.

Previous work focuses on semantic segmentation, but this paper not only focuses on instance detection, but in BEV. 

The output results actually does not look super impressive, but this provides a brand new direction for BEV perception.

#### Key ideas
- [DETR](detr.md) Use two sets of query vectors, one set for centerlines and one for objects.
- Lane branch
	- Detection head: if the lane encoded by the query vector exists
	- Control head: 2xR control points (R=3)
	- Association head: $\delta$-dimension vector
	- Association classifier: takes in $2 \delta$ dimension and judge whether they centerline pairs are associated. Input is $(N \times N) \times 2\delta$ dimension.
- Object branch
	- detection head: class probability distribution (including no detection)
	- 5-param head (x and y offsets, w, h, yaw)
- Split positional embedding
	- Half of the channel size is image space PE, and the other half is BEV space PE of that particular pixel (with GT transformation) --> This assumes a fixed camera extrinsics
	- The top half of BEV PE is not defined.
	- The PE is then added to the feature map
- Structured lane detection

#### Technical details
- Angle loss $L_{angle} = |\cos(2\alpha) - \cos(2\phi)| + |\sin(2\alpha) - \sin (2\phi)|$
- The road are represented in Bezier curves. Each curve has three control points. 
	- Bezier curves are a good fit for centerline since it allows us to model a curve of arbitrary length with a fixed number of 2D points. 
- The evaluation metrics of directed lane line graph
	- Precision/Recall: for lines that are matched, count the interpolated points TP
	- Detection ratio: how many line segments are matched/missed
	- Connectivity: fragmenting a GT into multiple connected segments is not an issue


#### Notes
- code on [Github](https://github.com/ybarancan/STSU)