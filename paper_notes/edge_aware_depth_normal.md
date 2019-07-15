# [Unsupervised Learning of Geometry with Edge-aware Depth-Normal Consistency](https://arxiv.org/pdf/1711.03665.pdf)

_July 2019_

tl;dr: Extend [SfM-Learner](sfm_learner.md) by introducing a surface normal presentation.

#### Overall impression
The idea is good, that we introduce a surface normal map, which at each point should be perpendicular to the depth estimation. 

However how to use it is a bit questionable. This work used normal map as an intermediate step (depth --> norm --> depth) and both conversion is deterministic by 3D geometry constraint. How this helps is puzzling. The result to be honest is not as good as claimed. You still see a lot of discontinuity of surface normals within the same object.

This work is superceded by their CVPR 2018 spotlight paper [LEGO](lego.md).

