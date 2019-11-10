# [Bayesian YOLOv3: Uncertainty Estimation in One-Stage Object Detection](https://arxiv.org/abs/1905.10296)

_November 2019_

tl;dr: Extension of the work on [towards safe ad](towards_safe_ad.md) to one-stage detector.

#### Overall impression
This paper is very similar to [Gaussian Yolov3](gaussian_yolov3.md) and models epistemic uncertainty as well.

Predicting uncertainty is critical for downstream pipelines such as tracking or sensor fusion. **Aleatoric** uncertainty captures sensor noise or ambiguities in the problem itself. **Epistemic** uncertainty is a measure for underrepresented classes in a dataset. Building a model ensemble is a way to estimate uncertainty.

They also observed that aleatoric scores correlates well with occlusion, but not epistemic. 

> However most of these methods have no measure of how certain they are in their output. When confronted with previously unseen data there is usually no way to measure if the model can deal with this input. For example a model trained on good weather data is faced with adverse weather situations.

> From the standpoint of active learning, label only the data with the most information gain could help cutting time and cost.

The paper also observes that Modeling aleatoric uncertainty boosted the performance (by 1-5%). Modeling epistemic uncertainty via monte carlo dropout degrades performance slightly. --> similar to [towards safe ad](towards_safe_ad.md).

#### Key ideas
- Add dropout layers to predict epistemic uncertainty. The dropout is applied immediately after conv, so it is OK to use BN layer afterwards. Using BN helps to use pertained model without dropout.
- There is a general trend that both aleatoric and epistemic uncertainty increases with decreasing IoU. This is similar to [gaussian yolov3](gaussian_yolov3.md) and a bit different from the findings from [towards safe ad](towards_safe_ad.md).

#### Technical details
- The paper showed that how the negative log-likelihood loss is essentially aleatoric uncertainty, with a trick to improve numerical stability. 
$$L = e^{-s} (y - f(x))^2 + s$$
- The s is also clipped to [-40, 40] to avoid NaN error.
- The paper uses one-stage detector and the results are from different feature maps. Thus the uncertainties are not combined but rather evaluated separately from each feature map and for each set of anchors. Or they need to be calibrated. --> Two stage detectors have a single output, making the evaluation of uncertainty easier. 


#### Notes
- [github link](https://github.com/flkraus/bayesian-yolov3)

