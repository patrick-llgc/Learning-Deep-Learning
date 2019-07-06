# [Deep Depth Completion of a Single RGB-D Image](https://arxiv.org/pdf/1803.09326.pdf)

_June 2019_

tl;dr: Two stage solution for depth completion problem: first stage use RGB to predict surface normals. Then use depth measurement as regularization for global optimization.

#### Overall impression
The paper proposed that surface normals are easier to predict as they do not scale with depth. The methods is superior to previous SOTA in predicting depth with RGB only image. 

The two stage pipeline is flexible but also slow. The advantage is that the first stage only depend on RGB and does not need to be retrained for different depth sensors. The second stage is a sparse optimization problem with depth constraint with real depth measurement data which can be solved on CPU (the inference time is ~1 sec thus slow).

[DeepLidar](deeplidar.md) proves that the depth can be done with an end-to-end solution. However it also converts lidar to image. Maybe we could combine the idea of DeepLidar with [PseudoLidar](pseudo_lidar.md).

#### Key ideas
- Biological evidence: estimating absolute depth from a monocular image is [difficult even for people](http://sci-hub.tw/https://doi.org/10.1007/BF00342882). However estimating surface normals (local differential properties) is [much easier](https://dspace.library.uu.nl/bitstream/handle/1874/7542/kappers_92_surface_perception_pictures.pdf?sequence=1).
- First stage two CNNs are used to predict surface normals and occlusion boundary from the color image. Both are local properties and can be estimated easily. The occlusion boundary is similar to the confidence map in deeplidar and is used to adapt for misalignments for color and depth measurements. 
- Second stage is global optimization using depth and depth consistency as constraints. As surface normals are used, the consistency of local tangents with surface normals are also included in the optimization objective.

#### Technical details
- The indoor depth completion and [outdoor depth completion](deeplidar.md) solves different problems. The challenge in indoor depth completion lies in glossy or bright surfaces where large areas of data is missing due to failure of depth sensor (Kinect or Intel RealSense). It is usually random sparsity. Outdoor depth completion is usually due to the nature of lidar sensor. It is usually structured sparsity.
- In indoor images, depth images often misses more than 50% of pixels (larger pixels than image).
- Mathematically it is insufficient to estimate depth using surface normals and occlusion boundary to estimate absolute depth. Pathological cases include the depth of a region in a window. However in real applications such cases are rare.

#### Notes
- Questions and notes on how to improve/revise the current work  

