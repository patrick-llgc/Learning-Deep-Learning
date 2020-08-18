# [Rethinking Pseudo-LiDAR Representation](https://arxiv.org/abs/2008.04582)

_August 2020_

tl;dr: Summary of the main idea.

#### Overall impression
The paper builds on top of the [Frustum-Pointnet](frustum_pointnet.md) version of pseudo-lidar, with a two-step process. First a 2D detector finds the car, and then crop a frustum with the bbox, then a [pointnet](pointnet.md) is used to place a 3D bbox around the point cloud. [PatchNet](patchnet.md) starts from this cropped patch.

The idea is simple, instead of RGB, fill in the XYZ values in the original image. It is similar to the idea of [CoordConv](coord_conv.md) and [CamConv](cam_conv.md), and perhaps should have been called 3DCoordConv. Along the line of work in lidar object detection, it is similar to the spherical or cylindrical view rendering of lidar point cloud into a raster image.

#### Key ideas
- **Data representation is not the key. Coordinate transformation is.** As long as the x, y, z are provided, they can be packed into a 2D image format. Instead of the RGB info, we can use XYZ. 
- The coordinate transformation implicitly encode camera calib information. --> what happens if we provide RGB + CoordConv + CamConv? Will the network be able to learn the transformation?
- PointNet-like architecture is not as mature and well studied as 2D conv. Using image based representation can leverage the advances in 2D conv. 
- This is corroborated with the idea that using cylindrical representation of lidar point cloud for object detection, such as in [Pillar OD](pillar_od.md).
- Feeding {u,v,z} leads to much worse results than feeding {x,y,z} directly.


#### Technical details
- The paper has a really solid approach to pick apart what works in a system. It reimplemented the pseudo-lidar as a baseline, first to compare with existing publication, and then to use as a foundation to build PatchNet. This is similar to [ATSS](atss.md).
- Masked global pooling to separate foreground and background.
- Uses multi-head structure to separate the difficult cases and easy cases. All heads do the same prediction simultaneously, only that we have to train a switcher to switch among them. It is a typical way to tackle **multi-modal distribution**.

#### Notes
- Why not concatenate RGB with XYZ, like [Pseudo-lidar Color](pseudo_lidar_color.md) does.

