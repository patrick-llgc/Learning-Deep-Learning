# [Waymo Dataset: Scalability in Perception for Autonomous Driving: Waymo Open Dataset](https://arxiv.org/abs/1912.04838)

_October 2020_

tl;dr: Waymo open dataset, a multimodal (camera, lidar) dataset covering a wide range of areas (SF, MTV, PHX).

#### Overall impression
This paper has a good review of all recently released datasets (Argo, [nuScenes](nuscenes.md), Waymo), except Lyft dataset.

#### Key ideas
- 1150 sequence each spanning 20 seconds. (nuScenes has 1000 sequence each spanning 20 seconds.)
- 2D tightly fitting bounding boxes for the camera images, and 3D bboxes. Note that this is different from 2D projection of 3D bboxes. 
- Hardware
	- Lidar
		- 5 lidars
		- Elongation: this is one additional attribute to lidar point cloud, similar to intensity. **High elongation with low intensity is strong indicator for spurious objects, such as dust, fog, rain**. Low intensity itself is not enough. 
	- Camera
		- 5 cameras, only front and side facing, each with horizontal FoV of 50 deg. No rear facing cameras. 
	- Synchronization: very good within +/- 7 msec. 
	- Rolling shutter projection: there are some quadratic optimization involved. --> Need to find more about this. Is this similar to motion compensation?
- Annotation and evaluation
	- Densely labeled, 10 Hz. (nuScenes only label at 2 Hz)
	- Difficulty: either specified by annotators, or < 5 within 3D bbox.
	- MOD is Evaluated by IoU, signs are evaluated by center distance. 
- Domain gap:
	- SF vs MTV/PHX
	- Not enough pedestrians in suburban areas
	- Cross domain learning: train on suburban/urban and test on urban/suburban areas. 
	
#### Technical details
- Two levels of labels: 
	- LEVEL 2 to examples where either the labeler annotates as hard or if the example has ≤ 5LiDAR points. 
	- the rest of the examples are assigned to LEVEL 1.

#### Notes
- [SemanticKITTI](semantic_kitti.md) provides point level segmentation.
- [Honda research institute 3D dataset](h3d.md) provides 160 crowded urban driving scenes. 
- [nuScenes](nuscenes.md) provides rasterized top-down semantic map, and [Argoverse](argoverse.md) provides detailed geometric and semantic maps, with a vector representation of road lanes and their connectivity.
- **Lidar Elongation**: Lidar elongation refers to the elongation of the pulse beyond its nominal width. Returns with long pulse elongation, for example, indicate that the laser reflection is potentially smeared or refracted, such that the return pulse is elongated in time. ([source](https://waymo.com/open/data/))
- **S2 cells**: Within Google we use a clever library built by Eric Veach to identify regions on the surface of the earth. It internally uses 64 bit numbers to uniquely identify cells, with sizes ranging from millimeters to hundreds of kilometers on a side, using projections of spheres onto the surfaces of cubes, quadtrees, and enumeration of positions on Hilbert space filling curves. This is one of my favorite libraries at Google. ([quote](https://pokemongohub.net/post/article/comprehensive-guide-s2-cells-pokemon-go/) from Jeff Dean). Pokemon GO seems to rely heavily on S2 cells. 