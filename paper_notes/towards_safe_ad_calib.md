# [Can We Trust You? On Calibration of a Probabilistic Object Detector for Autonomous Driving](https://arxiv.org/abs/1909.12358)

_November 2019_

tl;dr: Calibration of the network for a probabilistic object detector

#### Overall impression
The paper extends previous works in the [probabilistic lidar detector](towards_safe_ad.md) and its [successor](towards_safe_ad2.md). It is based on the work of Pixor. 

Calibration: a probabilistic object detector should predict uncertainties that match the natural frequency of correct predictions. 90% of the predictions with 0.9 score from a calibrated detector should be correct. Humans have intuitive notion of probability in a frequentist sense. --> cf [accurate uncertainty via calibrated regression](dl_regression_calib.md) and [calib uncertainties in object detection](2dod_calib.md).

A calibrated regression is a bit harder to interpret. P(gt < F^{-1}(p)) = p. F^{-1} = F_q is the inverse function of CDF, the quantile function.

> Unreliable uncertainty estimation in object detectors can lead to wrong decision makings in autonomous driving (e.g. at planning stage).

The paper also has a very good way to visualize uncertainty in 2D object detector.

#### Key ideas
- This paper adds aleatoric uncertainty for each regression target based on Pixor. 
- Calibration of classifiers
	- Empirical = P(label = 1 | pred = p)  = I(label = 1 and pred = p) / I (pred = p) = p = Theoretical
	- Bin prediction scores, and count empirical ones to plot the calibration plot
- Calibration of Regression
	- Empirical = P(label < F_q (p)) = I(label < F_q(p)) / N = p = Theoretical
- ECE (expected calibration error) is the weighted area of calibration plot and the diagonal line, N_m is the number of samples in the m-th interval.

$$ECE = \sum_i^M \frac{N_m}{N}|p^m - \hat{p^m}|$$

- Isotonic regression (保序回归)
	- During test time, the object detector produced an uncalibrated uncertainty, then corrected by the recalib model g(). In practice, we build a recalib dataset from validation data.
	- Post-processing, does not guarantee recalibration of individual prediction (only by bins). 
	- It changes probability distribution, Gaussian --> Non-Gaussian
	- Depends on the recalibration dataset. 
- Temperature scaling
	- One temperature per regressor (6 in total) to optimize the NLL score on recalibration dataset. 
	- Same distribution
	- Works best with only a tiny amount of data (0.5% of entire validation dataset)
- Calibration loss
	- loss between calculated $\sigma$ and regressed one
	$$L_{calib} = |\sigma - (pred - gt)|$$
	- Same distribution
- After recalibration, the confidence intervals are larger such that they fully cover the gt. 
- Generalization to new dataset requires a tiny amount of data (~1%) to generalize (from KITTI to nuScenes).

#### Technical details
- Achieving higher detection does not guarantee better uncertainty estimation
- Higher detection accuracy when trained with calibration loss

#### Notes
- The idea of proposing one bbox per pixel/point seems to have come from the PIXOR paper. Cf [LaserNet](lasernet.md) and [Point RCNN](point_rcnn.md).

