# [LiDAR R-CNN: An Efficient and Universal 3D Object Detector](https://arxiv.org/abs/2103.15297)

_September 2021_

tl;dr: Add proposal info to the point cloud before feeding into the 2nd stage.

#### Overall impression
PointNet could make the learned features ignore the size of proposals, as it only aggregates the features from points while ignoring the spacing in 3D space. The spacing encodes essential information such as the scale of the objects.

[Lidar RCNN](lidar_rcnn.md) provides a plug-and-play module to any existing 3D detector to boost performance. --> This could be useful for offline perception.

It is a point-based method, like [Point RCNN](point_rcnn.md) and [PV RCNN](pv_rcnn.md).

#### Key ideas
- Input features: include some contextual points around it (such as ground points).
- The pointNet architecture used to do the bbox refinement is extremely lightweight and fast. --> It runs quite fast on GPU, but is not efficient (due to MLP) in DLA.
- Size ambiguity problem
	- proposals of different size containing the same set of points will have exactly the same feature representation. However their cls and reg target may diff a lot.
- Two solutions
	- **Boundary offset**: every point has two new features, offset to the boundary, appended to them
	- **Virtual grid points**: grid points evenly distributed within the proposal, with binary bit indicating whether it is virtual or not.
	- **Other methods may also work as long as the proposal size information is fed into the 2nd stage**
- **Size normalization** works well if only one class is detected and regressed. In multi-class setting, the class ambiguity emerges if the size is ignored. --> they should work if only the bbox size and location (not the class) need to be refined.

#### Technical details
- Loss of orientation is not formulated as a multi-bin but a min of two losses (comparison with gt 180 apart). They are equivalent, as min can still facilitates grad flow and is still differentiable.
- Level-I and Level-II follows the definition of [Waymo Open Dataset](wod.md).

#### Notes
-[code on github](https://github.com/tusimple/LiDAR_RCNN)

