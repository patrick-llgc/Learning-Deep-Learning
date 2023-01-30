# [PPGeo: Policy Pre-training for End-to-end Autonomous Driving via Self-supervised Geometric Modeling](https://arxiv.org/abs/2301.01006)

_January 2023_

tl;dr: Large scale visual pretraining for policy learning.

#### Overall impression
The idea is interesting: how to use large scale pretraining to extract driving relevant information. The 

#### Key ideas
- Step1: use large scale uncalibrated driving video to train depthNet and poseNet, a la [SfMLearner](sfm_learner.md). The input is two consecutive frames at 1 Hz. 
- Step2: from a single image, predict the ego motion. --> This is highly questionable. It would be better to feed in multiple historical frames, and also historical ego motion information. If historical information is important for prediction tasks, why not for planning?

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work
