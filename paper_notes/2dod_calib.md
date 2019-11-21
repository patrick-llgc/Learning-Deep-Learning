# [Calibrating Uncertainties in Object Localization Task](https://arxiv.org/abs/1811.11210)

_November 2019_

tl;dr: Proof of concept by applying Uncertainty calibration to object detector. 

#### Overall impression
For a more theoretical treatment refer to [accurate uncertainty via calibrated regression](dl_regression_calib.md). A more detailed application is [can we trust you](towards_safe_ad_calib.md).

#### Key ideas
- Validating uncertainty estimates: plot regressed aleatoric uncertainty $\sigma_i^2$ and $(b_i - \bar{b_i})^2$
- To find 90% confidence interval, the upper and lower bounds are given by $\hat{P^{-1}}(r \pm 90/2)$, where $r = \hat{P(x)}$ and $\hat{P}$ is the P after calibration.

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

