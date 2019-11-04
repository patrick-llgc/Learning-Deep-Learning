# [IoUNet: Acquisition of 	Localization Confidence for Accurate Object Detection](https://arxiv.org/abs/1807.11590)

_November 2019_

tl;dr: Regress a separate branch to estimate the quality of object detection in terms of IoU. 

#### Overall impression
The vanilla version of IoU-Net (with the prediction of IoU and Precise RoI Pooling) is already better than baseline, most likely due to the regularization effect of the IoU branch. 

The authors then use this IoU score for IoU-guided NMS, and use the score estimator as a judge for iterative optimization.

#### Key ideas
- Summaries of the key ideas

#### Technical details
- Summary of technical details

#### Notes
- The idea of using a head to regress the IoU is very much like that of [FQNet](fqnet.md), although FQNet aims to regress the 3D IoU from overlaid wireframe. 

