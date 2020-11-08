# [The H3D Dataset for Full-Surround 3D Multi-Object Detection and Tracking in Crowded Urban Scenes](https://arxiv.org/abs/1903.01568)

_November 2020_

tl;dr: VoxelNet + UKF for 3D detection and tracking in crowded urban scene.

#### Overall impression
H3D dataset includes 160 scenes, and 30k frames, at 2 Hz. Roughly 90 seconds each scene. 

Really crowded scenes as H3D has roughly same number of people and vehicle.

#### Key ideas
- Use Lidar SLAM to register multiple lidar scans to form a dense point cloud. Then static objects will only have to be labeled once instead of in a frame-by-frame fashion.
- Camera is used to assist
	- Class annotation
	- 3D bbox verification after projection 3D bbox back to camera
- The 2Hz annotation is propagated to 10 Hz with linear velocity model.
- 3D detection with VexelNet and tracking with UKF. 

#### Technical details
- Calibration between GPS and Lidar are done with hand-eye calibration method. 
- Motion blur has to be corrected, using the method from LOAM. 


#### Notes
- Questions and notes on how to improve/revise the current work  

