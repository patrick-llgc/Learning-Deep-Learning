# [Vehicle Centric Monocular Velocity: End-to-end Learning for Inter-Vehicle Distance and Relative Velocity Estimation in ADAS with a Monocular Camera](https://arxiv.org/abs/2006.04082)

_July 2020_

tl;dr: Distance and velocity estimation from monocular video. 

#### Overall impression
Achieves better performance and is more end to end than [monocular_velocity](velocity_net.md). It uses optical flow and RoIAligned features to regress velocity and distance. It does not use off-the-shelf depth estimator as in [monocular_velocity](velocity_net.md).

3D velocity estimation can be seen as the prediction of sparse scene flow. This is to be compared to the 2d offset prediction in [CenterTrack](centertrack.md), which can be seen as a sparse optical flow. Scene flow = optical flow + depth.

SOTA velocity estimation is about 0.48 m/s.

#### Key ideas
- Input: two stacked images.
- **Main idea**: if we know the two corresponding point and their depth in two neighboring frames, then we can calculate the velocity of that point.
- Uses [PWCNet](pwcnet.md) encoder as feature extractor for feature F. 
- distance: feature vector F from RoIAligned current frame + geometry vectors (intrinsics + bbox)
	- Vehicle centric or not, does not matter much
- velocity: feature vector F + optical flow vector M RoIAligned from two neighboring frames + geometry vectors (intrinsics + bbox)
	- Velocity estimation needs to be vehicle centric as optical flow works much better on image patches than on the whole image.


#### Technical details
- It regresses the closest point to the vehicle, and uses bbox center as the proxy. This could be problematic for side distance estimation. 
- The supervised [DORN](dorn.md) performance is about the same as vehicle centric distance estimation. Self-supervised method is much worse. This is somewhat surprising.  

#### Notes
- Questions and notes on how to improve/revise the current work  

