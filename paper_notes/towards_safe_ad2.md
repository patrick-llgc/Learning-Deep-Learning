# [Leveraging Heteroscedastic Aleatoric Uncertainties for Robust Real-Time LiDAR 3D Object Detection](https://arxiv.org/abs/1809.05590)

_November 2019_

tl;dr: Model aleatoric uncertainty in both RPN and FRN.

#### Overall impression
This work extends previous work [towards safe ad](towards_safe_ad.md).

After modeling aleatoric uncertainty, performance is boosted almost 9%.

As we are modeling aleatoric uncertainty for each bbox, it is **hetereoscedastic** by nature.

However they did not report how prediction quality changes with predicted uncertainty.

#### Key ideas
- RPN generates 3D bbox that are axis-aligned. Then the RoIPooled features then regress four corners in BEV, top and bottom surface position, and encoded orientation (sin, cos).
- Aleatoric classification uncertainty is not explicitly modeled as it is self-contained from the softmax score. All regression loss are in the form of uncertainty aware loss.
$$L_{uncertainty} = e^{-t} L + t$$
- Train without uncertainty until almost converges, then add uncertainty. This trains faster.
- Modeling in FRH gives best performance in easy setting, and in both RPN and FRH gives best performance in moderate and hard. 
- Use TV (total variance) to quantify aleatoric uncertainty. $TV = \sum_i \sigma_i^2$
- Uncertainty findings:
	- The uncertainty grows with off-base axis angles
	- Uncertainty distribution has peak at smaller value compared to hard setting.
	- Uncertainty decreases with increasing softmax score --> This is a bit contradictory to previous findings. Maybe modeling uncertainty made softmax scores more representative to indicate uncertainty?

#### Technical details
- Summary of technical details

#### Notes
- Focal loss and online hard negative mining focuses on hard examples. However modeling uncertainty ignores noisy (and potentially hard) examples. How does a neural net distinguish this is a hard example or a noisy one?

