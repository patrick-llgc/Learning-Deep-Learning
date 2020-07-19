# [Kinematic 3D Object Detection in Monocular Video]()

_July 2020_

tl;dr: Mono3D with EKF to form temporally consistent tracks.

#### Overall impression
The paper is one of the first study in leveraging monocular video for 3D object detection (video-based 3d object detection). The study proposes several improvements over baseline [M3D-RPN](m3d_rpn.md). It is possible to predict the ego motion and object motion respectively.

The performance boost based on kinematics is not huge, but it makes the tracks temporally coherent. 

The EKF is a postprocessing module after the mono3D object detector.

KITTI datasets seem to provide **4 temporally adjacent frames** for each annotated frame. Kinematic mono3D uses 4 frames for inference.

#### Key ideas
- Decompose angle into axis, heading and offset. This is one step further to the multi-bin regression proposed by [deep3Dbox](deep3dbox.md).
	- The idea is that telling if a car is perpendicular is easier to tell if it is facing left or right. Thus the equally expressive 4-bin setting is reconfigured to 2-bin followed by another 2-bin. This **cascaded classification** boosts the 3D object detection performance AP_3D by almost 2%.
- Self-balancing 3D confidence
	- Predict a confidence score. If high, then use the 3D loss, otherwise use the average 3D loss within the same batch.
	- This is quite similar to the idea of [aleatoric uncertainty](uncertainty_bdl.md). It is the opposite to hard negative mining by encouraging the network to focus on reasonable examples.
- Ego motion is predicted by PoseNet. 
	- Regresses R and t separately, but uses an attention mechanism instead of only fc layers. 
- Motion model: extremely simplified, linear motion with constant velocity
	- constant size2	
	- constant heading direction
	- constant scalar velocity, can only move in the heading direction
- Kalman filter
	- EKF allows for use of real-world motion models as strong priors, and it is computationally efficient, and provides useful by-products. The uncertainty comes from the self-balancing 3D conf.
	- Forecasting: from $\tau_{t-1}$ to $\tau_t'$. Use transition matrix (only update x and y in the direction of v) and add ego motion on top of it.
	- Association: associate forecasted object state with observation (mono3D results at timestamp t). It has two stages: 3d distance based, then 2d iou based. 
	- Update: compute Kalman gain and update the tracklet
	- Ego motion predicted by neural network (like PoseNet)


#### Technical details
- Velocity estimation accuracy: 3.14 m/s for object velocity, and 2.89 m/s for ego motion.
- EKF can be used to perform forecasting as well. The accuracy drops further into the future.

#### Notes
- Questions and notes on how to improve/revise the current work  

