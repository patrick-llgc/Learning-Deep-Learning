# [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)

_December 2019_

tl;dr: Invented one simple yet effective method (temperature scaling) for CNN classifiers.

#### Overall impression
This paper is one of the first and most cited papers in calibration of neural networks. 

> In real world decision making systems, cls networks must not only be accurate but also should indicate when they are likely to be incorrect. 

#### Key ideas
- Modern neural nets are poorly calibrated (shallow MLPs are much better, e.g., LeNet).
- A single-variable version of Platt scaling, namely temperature scaling, is the most effective method at obtaining calibrated probabilities. 
- **Reliability diagrams (calib plot)** are drawn by dividing predictions into M bins, and calculate the accuracy and mean confidence (prediction) in each bin. This can be summarized into a scalar summary statistic of calibration
	- ECE (expected calibration error) is a weighted average of each bins' accuracy/confidence difference. --> see [towards safe ad](towards_safe_ad_calib.md).
	- MCE (maximum calib. error) in high-risk applications where reliability conf is absolutely necessary. 
	- NLL (neg. log likelihood) is also CE in ML context. 
- Miscalibration is related to increased model capacity (depth and width) and lack of regularization. Networks with BN is more likely to be miscalibrated. (weight decay is not used anymore in training modern nn, but it is useful in curbing miscalib.)
- **Neural nets can overfit to NLL without overfitting to 0/1 loss (accuracy)**. Overfitting to NLL is beneficial to cls accuracy. It learns a better cls acuracy at the expense of well modeled probabilities. --> **High capacity models are not necessarily immune from overfitting, but rather overfitting manifests in probabilistic error rather than cls error.**
- Calibration: requires a holdout validation set. 
	- Histogram binning: fixed bins, floating calib score (target proba)
	- Isotonic regression: floating bins and calib scores. Strict generalization of histogram binning
	- Platt scaling: feed uncalib scores into logistic regression. $q = \sigma(a z + b)$. Both a and b can be optimized using NLL loss over cvalidation set. 
- Extension to Multiclass:
	- Binning methods: K 1-vs-all problems. At test time, we get unnormalized probabiltiy $[\hat{q_i^{(1)}}, ..., \hat{q_i^{(K)}}]$. **This may change prediction results!**
	- matrix and vector scaling: KxK matrix or KxK diaganal matrix. 
	- temperature scaling: single parameters for all classes. This does not change prediction results. Infinite T means all equal proba 1/K, max uncertainty. Zero T means a very sharp delta prediction. During training, temperature increases monotonically, meaning that the network is getting more and more (over-)confident. 

#### Technical details
- Typical ECE for common cls datasets are between 4 to 10%.
- Temperature scaling works best shows that nn miscalib is intrinsically low dimensional.
- Latency: Temp scaling is also very fast, even faster than histogram binning or isotonic regression.

#### Notes
- Questions and notes on how to improve/revise the current work  

