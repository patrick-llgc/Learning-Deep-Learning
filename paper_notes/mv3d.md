# [Multi-View 3D Object Detection Network for Autonomous Driving](https://arxiv.org/pdf/1611.07759.pdf)

_Mar 2019_

tl;dr: sensor fusion framework to take in lidar point cloud and RGB images as input and predict oriented 3D bboxes. The 3D point cloud is encoded to a multi-view (birds eye view and front view) representation.

#### Overall impression
The paper is one of the pioneering work to integrate RGB and lidar point cloud data. It sets a good baseline in the task of 3D proposal geenration, 3D detection and 3D localization. Its performance is surpassed by [AVOD](avod.md), [F-pointnet](frustum_pointnet.md), [edgeconv](edgeconv.md), [point RCNN](point_rcnn.md) etc.

#### Key ideas
- Data preprocessing: lidar point cloud --> BV image with (M+2) channels, unrolled panorama FV image (projection on to a cylinder) with 3 channels. The point cloud encoding into a 2D pseudo image is hand-crafted fixed encoding scheme.
- The 3D object detector largely follows Faster RCNN pipeline: region proposal and second-stage refinement with fused features, each ROI-pooled from different modals.
- Bbox proposal generation from birds eye view of lidar point cloud. The 3D proposals are quite crude, axis-aligned 2D proposals with fixed height. Input is M+2 channels BV image, discretized with 0.1 m resolution.
	- Why birds eye view for region proposal? Preservation of physical sizes, and avoidance of occlusion.
- The proposal is converted to BV, FV and camera coordinate and ROI pooled features are fused to perform object classification and coordinate regression.


#### Technical details
- Regress coordinates for all 8 corners (24 points). This redundant representation performs better than other parameterization scheme. (Normally an oriented 3d bbox is paramerized as xyzwhl and heading angle theta)
- Point cloud is projected to a cylinder plane for front view image, with 3 channels height, distance and intensity.
- Regularization has drop path training and auxiliary loss. This seems to be a good strategy for multi-path training in general.
	- Drop path training: path level dropout
	- Auxiliary loss: each path in the auxiliary network share the weights with the original network, and only that path (without fusio with other views) is responsible for classification and bbox regression.
- 91% proposal recall for 300 proposals @0.5IOU (cf 91% recall with 50 proposal for AVOD, and 96% recall with point rcnn)
- If only a single view is taken as input, then birds eye view performs the best. 


#### Notes
- Q: can we fuse radar and camera data together? How does this affect sensor fusion (assumption of correlation)?
- Q: The deep fusion scheme is actually a special case of early fusion, with special handling with network training.

