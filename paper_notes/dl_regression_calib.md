# [Accurate Uncertainties for Deep Learning Using Calibrated Regression](https://arxiv.org/abs/1807.00263)

_November 2019_

tl;dr: Extends NN calibration from classification to regression.

#### Overall impression
The paper has a great introduction to the background of model calibration, and also summarizes the classification calibration really well.

The method can give calibrated credible intervals given sufficient amount of iid data.

For application of this in object detection, see [calibrating uncertainties in object detection](2dod_calib.md) and [can we trust you](towards_safe_ad_calib.md).

#### Key ideas
- For regression, the regressor H outputs at each step t a CDF $F_t$ targeting $y_t$. 
- A calibrated regressor H satisfies
$$\frac{1}{T}\sum_{t=1}^T\mathbb{I}\{y_t \le F_t^{-1}(p)\} = p$$ for all $p \in (0, 1)$. This notion of calibration also extends to general confidence intervals. 
- The calibration is usually measured with a calibration plot (aka reliability plot)
	- For classification, divide pred $p_t$ into intervals $I_t$, then it plots the predicted average x = $mean(p_t)$ vs empirical average y = $mean(y_t)$, for $p_t \in I_t$.
	- For regression, construct dataset 
$$\mathcal{D} =\{F_t(y_t), \frac{1}{T}\sum_{\tau=1}^T\mathbb{I}\{F_\tau(y_\tau) \le F_t(y_t) \} \}_{t=1}^T$$ 
As approximation, divide to bins $I_t$, for $p_t \in I_t$, plots the predicted average x = $mean(p_t)$, vs the empirical average y = $ \frac{1}{T}\sum_{\tau=1}^T\mathbb{I}\{F_\tau(y_\tau) \le p_t \}$. Then fit a model (e.g., isotonic regression) on this dataset.
	- For example, for p - 0.95, if only 80/100 observed $y_t$ fall below the 95% quantile of $F_t$, then adjust the 95% to 80%.

#### Technical details
- Evaluation: calibration error
$$CalErr = \sum_j w_j (p_j - \hat{p_j})^2$$
	- cf ECE (expected calibration error) from [can we trust you](towards_safe_ad_calib.md)

#### Notes
- [model calibration in the sense of cls](https://pyvideo.org/pycon-israel-2018/model-calibration-is-your-model-ready-for-the-real-world.html)
- Platt scaling just uses a logistic regression on the output of the model. See [this video](https://pyvideo.org/pycon-israel-2018/model-calibration-is-your-model-ready-for-the-real-world.html) for details. It recalibrates the predictions of a pre-trained classifier in a post-processing step. Thus it is classifier agnostic.
- [isotonic regression (保序回归)](https://scikit-learn.org/stable/auto_examples/plot_isotonic_regression.html) is a piece-wise constant function that finds a non-decreasing approximation of any function.

```python
ir = IsotonicRegression() # or LogisticRegression()
ir.fit(p_holdout, y_holdout)
p_calibrated = ir.transform(p_holdout)
```