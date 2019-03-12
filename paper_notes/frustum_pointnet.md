# [Frustum PointNets for 3D Object Detection from RGB-D Data](https://arxiv.org/pdf/1711.08488.pdf) (F-PointNet)
_Mar 2019_

tl;dr: Combines mature 2D object detection and advanced 3D deep learning for 3D object detection.

#### Overall impression
This paper builds on top of the seminal [pointnet](pointnet.md) for segmentation, and combines with 2D object detection to build a 3D  object detection pipeline.

#### Key ideas
- Data: Lift GRB-D scans to point clouds. Goal: estimation of oriented 3D bbox.
- Three steps in Frustum PointNet:
    - Frustum proposal: Extruding 2D bbox from image detectors and extract 3D bounding frustum.
    - 3D instance segmentation: binary classification (assumes only 1 object per frustum, this is rather similar to semantic segmentation). Takes in the 2D bbox class as additional features.
    - Amodal 3D bbox estimation: 
- A series of normalization is applied for PointNet input. Frustum rotation, mask centroid subtraction are critical, and t-net regression also helps quite a bit.
    - Frustum rotation: frustum samples have more similar XYZ distributions
    - Mask centroid subtraction: object points have smaller and more canonical XYZ
    - T-Net regression: finds the amodal bbox center
- **Bird's Eye View (BV)** proposal can be combined with RGB (camera) image proposals to further boost the performance. Even without camera image, point cloud itself can be used to perform proposal and 3D detection (although performance degrades without camera. BV proposal also makes the detection of some highly occluded object in camera easy.


#### Technical details
- PointNet takes in (x, y, z) coordinates and lidar intensity as input. Input size is (nx4).
- 3D Bbox has 8 DOF, but only 7 are considered (center 3 + size 3 + heading angle around z axis).
- The paper used a hybrid classification and regression formulation. First classify into predefined bins, then regress the residual. This works much better than directly regress a large range number. This discrete-continuous loss is called MultiBin Loss.

#### Notes
- Lidar point cloud data captures 3 degrees of dimension: Azimuth, height/elevation/altitude, and distance/radius (according to the spherical coordinate system).
- Q: How to calculate the IoU of oriented 3D bbox?
    - Intersection of rotated 2D bbox ([visualization](https://stackoverflow.com/questions/11670028/area-of-intersection-of-two-rotated-rectangles/11672022)). Here is [the exact solution](https://stackoverflow.com/a/45268241). You can perform [sequential cutting technique](https://stackoverflow.com/a/11672022). Or you can find the vertices of the resulting polygon (they can only be the existing vertices of the rectangles or from intersections of the edge line segments). The area of the resulting polygon can be calculated with cross product.
    - A faster but less accurate way to do this is through rasterization.
    - code is [here](https://github.com/kujason/wavedata/blob/c4b5aabd9eb3b74fad777349f75161032d3460fa/wavedata/tools/obj_detection/evaluation.py) using rasterization.
- Resolution of depth image is still lower than RGB image from commodity cameras
- Camera projection matrix needed to register 3D point cloud with 2D RGB image and to extract a 3D frustum from the point cloud.
- Q: The paper combines the estimated 3D bbox from RGB image and BV image with 3D NMS. How about perform intersection of the frustum and BV cuboids before segmentation?
- Q: If T-Net regresses the residual from centroid of the segmented point cloud points to the true amodal bbox center, what does box-net regress? In particular, how to define their loss?
- KITTI provides synchronized RGB images and LiDAR point clouds. The model are trained on each frame assuming independence among frames.