# [Towards Safe Autonomous Driving: Capture Uncertainty in the Deep Neural Network For Lidar 3D Vehicle Detection](https://arxiv.org/abs/1804.05132)

_November 2019_

tl;dr: One of the first paper to introduce epistemic and aleatoric uncertainty in object detection.

#### Overall impression
This paper has a good level of details regarding how to adapt aleatoric and epistemic uncertainty to object detector. 

**Modeling aleatoric uncertainty boosted the performance (by 1-5%). Modeling epistemic uncertainty via monte carlo dropout degrades performance slightly.** --> also observed by [Bayesian yolov3](bayesian_yolov3.md).

This paper only models aleatoric uncertainty in FRH (faster rcnn head) part. This work is extended by [towards safe ad2](towards_safe_ad2.md) by modeling uncertainty in both RPN and FRH.

> Uncertainty can be used to efficiently improve the vehicle detector in an active learning paradigm: the detector actively queries the unseen samples with high epistemic uncertainty.

#### Key ideas
- Vehicle probability and **epistemic** classification uncertainty: N forward passes yields s_i (i=1,...,N) (N=40 in this paper)
	- proba: $$\frac{1}{N} \sum_i s_i$$
	- SE (Shannon entropy): 
$$SE = -(\frac{1}{N}\sum_i s_i) \log (\frac{1}{N} \sum_i s_i) - (1-\frac{1}{N} \sum_i s_i) \log(1-\frac{1}{N} \sum_i s_i)	 $$
	Note that this only depends on the average score
	- MI (mutual information)
$$MI = -(\frac{1}{N} \sum_i s_i) \log (\frac{1}{N}\sum_i s_i) + \frac{1}{N}\sum_i (s_i \log s_i + (1-s_i)\log(1-s_i))$$
- classification **aleatoric** uncertainty: not implemented
- 3D bbox and **epistemic** spatial uncertainty
	- Mean $$I=\frac{1}{N} \sum_i v_i$$
	- epistemic uncertainty in terms of total variance
$$C(x)=\frac{1}{N}\sum_i v_x v_x^T - I_x I_x^T$$
$$TV(x) = trace(C(x))$$
- 3D bbox **aleatoric** uncertainty
	- directly regress as additional number (uncertainty aware L1/L2 loss, similar to [KL loss](kl_loss.md))
$$L = e^{-\lambda} ||v_{gt} - v_{pred}|| + \lambda$$
	- the covariance matrix of the Gaussian multivariate is diagonal.
- MI/SE both correlates well with IoU (prediction quality) --> this is interesting, cf [IoU Net](iou_net.md). Maybe plotting the average in IoU will also reveal similar trends in IoU Net.
- TV/**epistemic** uncertainty decreases for higher IoU. Aleatoric uncertainty also goes down but not as much with prediction quality. --> cf [bayesian yolov3](bayesian_yolov3.md) from the same authors.
- The aleatoric uncertainty is positively correlated with **distance**. A more distant object is more difficult to localize, due to more sparse measurement. The same holds true for **occluded** ones.

#### Technical details
- Faster RCNN for BEV detection (with 0.1 m as a pixel) + fixed height to lift to 3D
- 3D bbox representation: 24 numbers of the 8 corners of 3d bbox, corner loss of distance from the ground truth, normalized by diagonal distance. 

#### Notes
> Knowing what an object detection model is unsure about is of paramount importance for safe autonomous driving.
> Most Object detection can only tell the human drivers what they have seen, but not how certain they are about it.
> Detecting an abnormal object that is different from the training dataset may result in high epistemic uncertainty, while detecting a distant/occluded object may result in high aleatoric uncertainty.

- Towards faster estimation of epistemic uncertainty, we can cache the feature map from the backbone and perform several forward pass only on the last FC layers. 
- Before we change our object detector to a probabilistic one, how good is the correlation between cls score and IoU? And evaluate this again after modeling aleatoric. 
