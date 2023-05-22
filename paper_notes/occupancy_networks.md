# [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/abs/1812.03828)

_May 2023_

tl;dr: Encoding the occupancy of a scene with a neural network, and can be queried at any location with arbitrary resolution.

#### Overall impression
In 3D there is no canonical representation which is both computationally and memory efficient yet allows for representing high-resolution geometry
of arbitrary topology.

Occupancy networks implicitly represent the 3D surface as the continuous decision boundary of a deep neural network classifier. Instead of predicting a voxelized representation at a fixed resolution, the network can be evaluated at arbitrary resolution. This drastically reduces the memory footprint during training.

#### Key ideas
- Training involves random sampling points inside the volume. Random sampling yields the best results. 
- During inference, a Multiresolution IsoSurface Extraction (MISE) method is used to extract isosurface of a scene. 

![](https://rlschuller.github.io/onet_site/img/common_arch.svg)

#### Technical details
- Summary of technical details

#### Notes
- [Official website with talk and supplementary materials](https://avg.is.mpg.de/publications/occupancy-networks)
