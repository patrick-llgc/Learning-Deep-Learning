## [Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite](http://www.cvlibs.net/publications/Geiger2012CVPR.pdf)
## [Vision meets Robotics: The KITTI Dataset](http://ww.cvlibs.net/publications/Geiger2013IJRR.pdf)

_Mar 2019_

tl;dr: KITTI vision benchmark suite, and data preprocessing.

#### Overall impression
The KITTI dataset underwent quite a lot of preprocessing, including rectification (for stereo vision tasks), calibration, synchronization.

#### Key ideas
- Equipment
	- 2 pairs of color/grayscale camera, 54 cm apart. It operates at 10 Hz, triggered when lidar heads forward. 
	- Velodyne HDL-64E (10 Hz), 2 cm distance accuracy, 0.1 degree angular resolution.
	- GPS/IMU localization unit with RTK correction
- Tasks
	- Stereo matching and optical flow: images are rectified from 1392x512 to 1240x376.
	- 3D visual odometry/SLAM: evaluating 6D poses. For a given set of image pairs, compute the relative difference of the GT difference and prediction difference of 6D poses. 
	- 3D vision: annotate tracklets in 3D space and project them back to image. Displays lidar point cloud as well as the camera images to increase the quality of annotation.
- A subset of data is selected to evaluate each task. 
	- For 3D object detection, images are iteratively added to the subset according to non-occluded objects as well as entropy of the object orientation distribution.
	- Image form one sequence does not appear in both training and test set.

#### Technical details
- GNSS (GPS, GLONASS, Beidou, Galileo), IMU and RTK correction signal. RTK (real time kinematics) is using the phase of the carrier wave to correct location signal and has a localization error of 5cm. The GNSS signal itself without RTK correction can only localize to a few meters.
- Camera images are calibrated with **intrinsic and extrinsic** parameters, with checkerboard image mounted on walls. 
- Velodyne to camera calibration is done semi-manually.


#### Notes
- Camera calibration
	- Intrinsic camera calibration: mainly find the **Lens distortion**.
	- Principal point in a pinhole camera model: intersection of optical axis on the image plane. Lens distortion calibration: may be separated from remaining calibration processes.
	- Radial lens distortion (simplified rule). Pu = f(Pd, rd), where rd is the distance of Pd to the principal point. 
	- Generally we use more points than equations for calibration, thus overdetermined. 
- A 3D rigid body transformation has 6DoF and is called a pose. A pose belongs to special euclidean group SE(3). For common representations of a pose, see review paper [here](http://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf).
- [Image rectification](https://en.wikipedia.org/wiki/Image_rectification): image processing so that the epipolar lines are horizontal (i.e., the corresponding points can only be in the same row in the other image). After rectification, the images are as if taken from two co-planar cameras. See [example image](https://images.slideplayer.com/32/9802393/slides/slide_14.jpg).
	- Epipolar line: in stereo vision, the corresponding point in the other image can only lie in a line, this line is named epipolar line. If two images are coplanar, i.e. they were taken such that the right camera is only offset horizontally compared to the left camera (not being moved towards the object or rotated), then each pixel's epipolar line is horizontal and at the same vertical position as that pixel. However, in general settings (the camera did move towards the object or rotate) the epipolar lines are slanted.

- [Pinhole camera model](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html). 
	- **Distortions** consist of mostly radial distortion and slight tangential distortion.  It is usually determined by a [distortion coefficient matrix](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html) of shape (1x5), k1, k2, p1, p2, k3. ki characterizes radial distortion (typically k1 and k2 are enough for non-fisheye lens) and pi characterizes tangential distortion. [Radial Distortion](https://en.wikipedia.org/wiki/Distortion_(optics)) can take on different forms: barrel (for wide angle lens), pincushion (for telephoto lens), or complex mixture of the two.
	- **Intrinsic** parameters have 4 DoF, principal points (cx, cy) and focal length (fx, fy). The matrix of intrinsic parameters does not depend on the scene viewed. So, once estimated, it can be re-used. Usually it is written in a 3x3 camera matrix. 
	- **Extrinsic** parameters have only 6 DoF, but usually uses 12 parameters in a 3x4 matrix [R|t].
- Uber has visualization tool [**AVS**](https://eng.uber.com/avs-autonomous-vehicle-visualization/), Open Standard for Autonomous Vehicle Visualization.
