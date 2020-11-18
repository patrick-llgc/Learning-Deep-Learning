# [MVF: End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds](https://arxiv.org/abs/1910.06528)

_November 2020_

tl;dr: Improve point embedding with dynamic voxelization and multiview fusion.

#### Overall impression
Both [VoxelNet](voxelnet.md) and [PointPillars](point_pillars.md) uses [PointNet](pointnet.md) to learn point embeddings, and generate pseudo-3D volume or pseudo-2D image to use 3D and 2D convolution. This paper improves the point embedding process by aggregating multiple views, and is a plug-and-play module that can be integrated into pointpillars. 

#### Key ideas
- **Dynamic voxelization**: instead of having a fixed array (K max voxel x T max point per voxel x F feat dim), it maintains a dictionary of the point-voxel relationship. 
	- The points in each dynamic voxel (having varying num of points) are aggregated into a fixed length feature by maxPooling, similar to PointNet. 
	- This avoid info loss and saves memory
- **Multiview fusion**: voxelization in Cartesian (X x Y x Z) with Z as the channel dim, and in speherical view (elevation x azimuth x radius) with radius as the channel dim. 
	- The features are extracted with a convolution tower that maintains the tensor shape
	- The features from Cartesian path, spherical path and original encoded features are concatenated. --> this enhances the original encoded features.
- The enhanced point features can be used in structures like [VoxelNet](voxelnet.md) and [PointPillars](point_pillars.md).

#### Technical details
- Dynamic voxelization can bring some KPI improvement (avoid info loss), but multiview fusion brings much more. 
- The lookup table is not readily available and need customized implementation of CUDA kernel to reach good speed. 

#### Notes
- Questions and notes on how to improve/revise the current work  

