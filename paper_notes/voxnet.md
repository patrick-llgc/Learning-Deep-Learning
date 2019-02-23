# [VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition](https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf)

_Feb 2019_

tl;dr: Convert range data such as point cloud and RGBD image to volumetric data, and perform 3D recognition.

#### Overall impression
Converting the point cloud data (sparse) into 3D volumetric data (dense) may benefit from the progress people have made in 3D image process, including **video recognition** and **medical image** analysis. Computationally it may not make sense. The 3D CNN is quite simple. It is very effective in classification but may not be ideal for semantic segmentation and object detection tasks.

#### Key ideas
- Point cloud vs Volumetric data
	- Convert range data into volumetric representation (it distinguishes free space vs unknown space). How? From 3D raytracing.
	- Spatial neighborhood queries in point cloud may quickly become intractable when the total number of points are large.
	- RGBD data can also be converted to point cloud once the intrinsic parameters of the scanner is known. RGBD has regular grid, and point cloud is more sparse.
- **Occupancy grid** is used as the volumetric representation of the range data. It is richer than point cloud as it distinguishes free (voxels a lidar beam passed through) and unknown space (voxels behind where a lidar beam hit).
- 3D CNN is not rotational invariant. Rotation is used in training as augmentation strategy, and used in inference as voting strategy.
- Multi-resolution input: a context stream that learns features on low-resolution frames and a high-resolution fovea stream that only operates on the middle portion of the frame. This idea comes from [a video recognition paper](https://cs.stanford.edu/people/karpathy/deepvideo/deepvideo_cvpr2014.pdf).


#### Technical details
- Three occupancy models are proposed, binary grid, density grid and hit grid. The first two need raytracing, and **hit grid** does not distinguish between the unknown and free space. The hit grid model works surprising well, and is fast as it does not need raytracing.
- For lidar data, spatial resolution is fixed at (0.1 m)^3 and (0.2 m)^3. This maintains the information given by the relative scale of objects. 

#### Notes
- More on occupancy grid
	- [wikipedia](https://en.wikipedia.org/wiki/Occupancy_grid_mapping)
	- [Notes from CMU](http://www.cs.cmu.edu/~16831-f14/notes/F14/16831_lecture06_agiri_dmcconac_kumarsha_nbhakta.pdf)
	- Sebastian Thrun is an expert on this topic. 


