# [Part-level Car Parsing and Reconstruction from a Single Street View](https://arxiv.org/abs/1811.10837)

_January 2020_

tl;dr: Train with synthesized data with parts and weekly labeled real data to transfer part knowledge. Directly regress distance.

#### Overall impression
The main problem this paper is trying to solve is occlusion. 

> Instance mask can also be ambiguous caused by object symmetries. Two similar masks could have quite different orientations.

It's hard to detect accurate landmarks for low-resolution cars. 

The model has 70 semantic parts, but during training, they are grouped into 13 super-parts, which is more reasonable.

#### Key ideas
- 3D annotation propagation: Annotate 40 models and use dense correspondence to transfer that to more than 300 models. First align with icp (iterative closest point) for rigid registration of point cloud (translation, rotation and scaling). Second apply embedded deformation (elastic registration) to register these cars.
- Each car is decomposed into 70 exterior part. --> This is way to many! This even include fuel door, door handle and logo.
- 3D shape space: each car has N point. PCA divide into Nx22 dimensional shape. Each car can be estimated with 22 shape coefficient. --> Compare with [3D RCNN](3d_rcnn.md) (PCA) and [RoI10D](roi10d.md) (auto encoder).
- Directly regressing distance from RoIPooled feature is an ill-posed problem (see [3D RCNN](3d_rcnn.md) for details). However augmented with the part features, in particular the part coordinates (similar to coord conv), they are able to regress distance directly. --> This is similar to concat bbox coordinates to the features. 
- Loss summary
	- direct loss: 2D proj of 3D center, shape param L1 loss, distance multibini loss, angle multi-bin loss
	- 3D loss: 3D shape loss via N-point vertices distance (vertices computed from shape params), 3D center loss via Rc*t (Rc is azimuth rotation), Rotated N-point vertices distance.
- With parts info, the performance on ApolloScape jumped from 17 to 23 points. Adding  3D loss also improved performance.

#### Technical details
- Feature encoder trained for car pixel classification (semantic segmentation) cannot be used for parts cls, but that trained for part cls can be used for part cls. This means that features learned at finer grained level can be used as feature at higher hierachical level, which is not true on the contrary.
- Instead of GAN, this paper used implicit knowledge transfer by semantic segmentation.

#### Notes
- Questions and notes on how to improve/revise the current work  

