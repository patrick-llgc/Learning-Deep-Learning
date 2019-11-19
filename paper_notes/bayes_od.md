# [BayesOD: A Bayesian Approach for Uncertainty Estimation in Deep Object Detectors](https://arxiv.org/abs/1903.03838)

_November 2019_

tl;dr: Fuse aleatoric and epistemic uncertainty into bayesian framework.

#### Overall impression
Very statistical paper. Lots of math details. The implementation in tensorflow probability (tfp) package. It reinterprets the DL object detector as one measurement device, with variance. 

The anchor boxes are different measurement devices. 

Marginalization over anchor distribution seem to predict epistemic uncertainty

**Open-set detection** problem (detecting unknown unknowns) is still one area safety-critical applications such as autonomous driving needs to focus on.

#### Key ideas
- The baseline: loss attenuation
$$L(x) = \frac{1}{2\sigma^2} |y - f(x)| + \frac{1}{2} \log \sigma^2$$
	- Note that the uncertainty goes with anchor. (in the original OD, each anchor predicts a bbox. Again, each anchor is a sliding window)
	- intelligent robust regression loss, more robust against outliers (noise)
	- second term serve as a regularizer
- Bayesian inference to replace NMS: 
	- greedy clustering per-anchor outputs with IoU thresh = 0.5 --> this can be improved by [Soft NMS](soft_nms.md)
	- all detection contributes in one cluster contribute to the final results, with contribution proportional to 1/uncertainty. --> this is very similar to the variance voting scheme proposed by [KL Loss](kl_loss.md)
- Aleatoric uncertainty estimation also includes a term for FP term. 
$$L = \frac{1}{N_{po}}\sum_i^{N_{pos}} L(x) + \frac{1}{N_{neg}}\sum_i^{N_{neg}} \frac{1}{\sigma(x)^2}$$
	- the second term is found to provide better numerical stability (but this did not transfer well to TF2.0 [as claimed by the author](https://github.com/asharakeh/bayes-od-rc/issues/2))

#### Technical details
- Black box (MC drop out) method has little discriminative power between TP and FP, mainly because it observes after NMS
- Sampling free (aleatoric regression) method are faster than black box ones (one forward pass instead of multiple). The authors say they provides worse categorical discrimination, but Fig. 3 seems to be contradicting this --> why?
- The authors used GMUE instead of Total variance (trace of the covar matrix). --> why?

#### Notes
- The number of object instances in the scene is unknown a priori --> maybe this is one area DL is better than deterministic optimization algorithms such as MUSIC?

