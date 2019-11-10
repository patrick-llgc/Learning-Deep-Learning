# [Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving](https://arxiv.org/abs/1904.04620)

_November 2019_

tl;dr: A probabilistic object detector with NLL loss to maximize the probability of labels.

#### Overall impression
This paper builds on top of YOLOv3 and makes each of the x, y, w, h two regression target, one mean and one stdev. 

> FP from a false localization (phantom tracks) during autonomous driving can lead to fatal accidents and hinder safe and efficient driving.

Adding the uncertainty prediction helps with TP and reduce FP dramatically. 

The NLL loss essentially models aleatoric uncertainty.

#### Key ideas
- The localization uncertainty indicates the relibility of bbox. Objectness score does not reflect the reliability of the box well.
- Bbox score = objectness x class_score x (1 - uncertainty_aver)
- IoU values tend to increase as localization uncertainty decreases on both KITTI and BDD datasets.
![](https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_225489%2Fproject_350848%2Fimages%2Fx4.png)

#### Technical details
- Loss is designed as NLL (negative log likelihood) of a Gaussian distribution. This helps to reduce sigma for confident predictions and increase sigma for non-confident predictions. --> This seems to be a bit different from what [Alex Kendall's uncertainty loss](uncertainty_bdl.md), but both loss learns to attenuate loss for non-confident predictions.

#### Notes
- The implementation is with darknet and the gradients have to be manually calculated and coded. Cf. [bayesian yolov3](bayesian_yolov3.md) for a tensorflow implementation of yolov3.

