# [PanopticBEV: Bird's-Eye-View Panoptic Segmentation Using Monocular Frontal View Images](https://arxiv.org/abs/2108.03227)

_October 2021_

tl;dr: Use of a vertical and a flat transformer to lift image into BEV.  

#### Overall impression
The paper notes correctly that the notion of instance is critical to downstream. [FIERY](fiery.md) also extends the semantic segmentation idea to instance segmentation. [Panoptic BEV](panoptic_bev.md) goes one step further and does [panoptic segmentation](panoptic_segmentation.md).

#### Key ideas
- Backbone: resnet + [BiFPN](efficientdet.md)
- At each level (P2-P5), image features are projected into a BEV by a dense transformer module (note that this transformer is not attention based).
- Dense transformer module
	- Each dense transformer consists of a distinct vertical and flat transformer.
	- The vertical transformer uses a volumetric lattice to model the intermediate 3D space which is then flattened to generated the vertical BEV features. --> this vertical transformer is quite similar to that of [Lift Splat Shoot](lift_splat_shoot.md).
	- The flat transformer uses IPM followed by an Error Correction Module (ECM) to generate the flat BEV features.
	- The differentiation between flat and vertical module is by a binary semantic segmentation network.

#### Technical details
- Summary of technical details

#### Notes
- [Video](https://www.youtube.com/watch?v=HCJ1Hi_y9x8)

