# [Deep MANTA: A Coarse-to-fine Many-Task Network for joint 2D and 3D vehicle analysis from monocular image](https://arxiv.org/abs/1703.07570)

_October 2019_

tl;dr: Predict keypoints and use 3D to 2D projection (Epnp) to get position and orientation of the 3D bbox. 

#### Overall impression
This is one of the first papers on mono3DOD. It detects 2D keypoints, and solves for the best position through 2D/3D matching.

This study reiterate the idea that **3D vehicle information can be recovered in monocular images because vehicles are rigid bodies with well known geometries.**

I feel that the 2D/3D matching is time consuming and discards lots of useful information that can be directly regressed from the image. 


#### Key ideas
- Architecture:
	- Step 1: cascaded RCNN to regress 2d bbox, classification, 2D keypoints, visiblity, template similarity
	- Step 2: with template simliarity, the best matching 3D CAD model is selected, and 2D/3D matching is performed to recover the 3D position and orientation.
- The 3D matching step uses both intrinsics and extrinsics information.
- It uses a semi-automatic way to label keypoints. It needs the 3D bbox gt (from lidar pipeline) and fit the CAD model in that location. Then 3D keypoints are projected to 2D images to form gt.

#### Technical details
- 103 CAD model x 36 keypoints --> this is significantly reduced by [monoGRnet 2](monogrnet_russian.md) to 5 CAD x 12 keyppints. 
- The network predicts the template similarity. This is similar to later methods that regress the offset from subclass averages (3D vehicle size cluster).

#### Notes
- Questions and notes on how to improve/revise the current work  

