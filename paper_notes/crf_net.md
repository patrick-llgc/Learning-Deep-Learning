# [CRF-Net: A Deep Learning-based Radar and Camera Sensor Fusion Architecture for Object Detection](https://ieeexplore.ieee.org/abstract/document/8916629/) 

_January 2020_

tl;dr: Paint radar as a vertical line and fuse it with radar. 

#### Overall impression
This is one of the first few papers that investigate radar/camera fusion on nuscenes dataset. It is inspired by [Distant object detection with camera and radar](distant_object_radar.md).

Both [CRF-Net](crf_net.md) and [Distant radar object](distant_object_radar.md) transforms the unstructured radar pins to pseudo-image and then process it with camera. Alternative approach to unstructured radar pins (point cloud) is to use [PointNet](pointnet.md), but PointNet is usually the best in classification or semantic segmentation when the RoI is extracted. Pseudo-image method is used in many other works, [PointPillars](point_pillars.md) for lidar data, and many work that incorporate intrinsics.

Embedding meta data info into conv:

- [meta data fusion for TL2LA](deep_lane_association.md)
- [fusing radar pins with camera](distant_object_radar.md)
- [cam conv](cam_conv.md) to fuse focal length into convs.
- [camera radar fusion net](crf_net.md)

#### Key ideas
- The architecture is RetinaNet with VGG, with radar fed in from multiple levels.
- Paint radar point as vertical line. Line starts from ground and extends 3 meters, and are thus not uniformly painted vertically. cf [Parse Geometry from a Line](depth_from_one_line.md).
- Accumulate radar in the past 13 frames (~ 1s) for more data
- Radar information include RCS and distance.

#### Technical details
- Training using BlackIn, essentially input dropout. Similar technique is used in [Qualcomm's radar camera early fusion](radar_camera_qcom.md) as well to increase the robustness of the network.
- It removes radar pins outside of 3D GT bbox. This removes a lot of noise in radar results.

#### Notes
- Background

> In heavy rain or fog, the visibility is reduced, and safe driving might not be guaranteed. In addition, camera sensors get increasingly affected by noise in sparsely lit conditions. The camera can also be rendered unusable if water droplets stick to the camera lens.

> Filtering out stationary radar object is common, but this may filter out cars under traffic light or bridges.

- [pdf backup](https://www.dropbox.com/s/5iftol6j5oq6t16/crf_net.pdf?dl=0)
- [github code](https://github.com/TUMFTM/CameraRadarFusionNet)

