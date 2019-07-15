# [LEGO Learning Edge with Geometry all at Once by Watching Videos](https://arxiv.org/pdf/1803.05648.pdf)

_July 2019_

tl;dr: Build on [SfM-Learner](sfm_learner.md) and added multi-task learning of edges and surface normals and self-consistency between the tasks. SOTA for static scenes.

#### Overall impression
The general idea is inherited from sfm-learner that it also uses view synthesis as supervision. However this work also predicts surface normals and edges, and added quite a few losses.

The key motivation of the work is that the estimated depth and surface normals are blurry and there are discontinuities inside a smooth surface. So the paper proposed a strong "as smooth as possible in 3D" prior, that all pixels should lie in the same planar surface if no edges exists in-between.

Seems that lots of technical implementation details to make this work. This paper is tightly coupled with their previous work at AAAI2018 [Unsupervised Learning of Geometry with Edge-aware Depth-Normal Consistency](edge_aware_depth_normal.md), but the AAAI work is not well written up. The AAAI idea is basically enforce a normal and depth consistency, and the edges are obtained using CV methods instead of jointly learned.

This paper also assumes a static scene and does not work well with occlusion and dis-occlusion.

#### Key ideas
- Geometric edge: mid-level edge (different from low-level internal edges from the same surface).
- The edges and smoothness are adversarial in the training pipeline. To minimize loss, more edges have to be produced to mask out loss, so a BCE regularization term is also introduced.
- From the 3D-ASAP prior, we have two new losses, one regularizes the depth map and the other regularizes normal map.
- The idea behind regressing surface norms is similar to that of [deepLidar](deeplidar.md) that estimating absolute depth from a monocular image is [difficult even for people](http://sci-hub.tw/https://doi.org/10.1007/BF00342882). However estimating surface normals (local differential properties) is [much easier](https://dspace.library.uu.nl/bitstream/handle/1874/7542/kappers_92_surface_perception_pictures.pdf?sequence=1).

#### Technical details
- Double edge issue: the depth loss from by 3D-ASAP prior always have a positive and a negative edge at the boundary. In order to minimize loss, the edge map is learned to have double edges at boundary to minimize loss. To fix this issue, only the positive edge is kept in the loss term.

#### Notes
- [spotlight video](https://youtu.be/WrEKJeK-Wow?t=4629)
- [code on github](https://github.com/zhenheny/LEGO) star is not many, and maybe the idea is not as practical as later work such as [monodepthv2](https://github.com/nianticlabs/monodepth2).
- 3D-ASAP: If there are no edges between two points, they should be in the same planar surface. This assumption works well for static scenes such as road and walls in urban driving scenario. --> This may be improved as the assumption is quite coarse.

