# [BEV-IPM: Deep Learning based Vehicle Position and Orientation Estimation via Inverse Perspective Mapping Image](https://ieeexplore.ieee.org/abstract/document/8814050)

_October 2019_

tl;dr: IPM of the pitch/role corrected camera image, and then perform 2DOD on the IPM image. 

#### Overall impression
The paper performs 2DOD on IPM'ed image. This seems quite hard but obviously doable. The GT on BEV image seems to come from 3D GT, but the paper did not go to details about it. 

The detection distance is only up to ~50 meters. Beyond 50 m, it is hard to reliably detect distance and position. --> Maybe vehicle yaw are not important for cars beyond 50 meters after all?

#### Key ideas
- Motion cancellation using IMU (motion due to wind disturbance or fluctuation of road surface)
- IPM assumptions:
	- road is flat
	- mounting position of the camera is stationary --> motion cancellation helps this.
	- the vehicle to be detected is on the ground
- 2DOD oriented bbox detection based on YOLOv3. 

#### Technical details
- KITTI does not label bbox smaller than 25 pixels, which translates to 60 meters according to fx=fy=721 of KITTI's intrinsics.

#### Notes
- [youtube demo](https://www.youtube.com/watch?v=2zvS87d1png&feature=youtu.be) the results look reasonably good, but how about occluded cases?

