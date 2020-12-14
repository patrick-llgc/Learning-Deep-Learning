# [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977)

_June 2019_

tl;dr: A new DL framework to deal with two uncertainties (aleatoric and epistemic) simultaneously. It learns the mapping from the input data to aleatoric uncertainty from the input data, without the need for explicit "uncertainty labels".

#### Overall impression
The paper summarizes nicely existing literature in the field about aleatoric and epistemic uncertainty. This is indeed an important area that can contribute to explainable AI.

#### Key ideas
- Two main kinds of uncertainty
	- Epistemic uncertainty
		- Measures what your model doesn’t know
		- Can be explained away by unlimited data
		- Data issue
		- Model uncertainty
	- Aleatoric uncertainty
		- Measures what you can’t understand from the data
		- Can be explained away by unlimited sensing
		- Sensing issue
		- Data uncertainty
- Applications:
	- Aleatoric uncertainty is important for
		- Large data situation (uncertainty is explained away with big data)
		- Real time application (no expensive MC experiments)
	- Epistemic uncertainty is important
		- Safety-critical applications
		- Small datasets (or on how to fill in corner case gaps)
- Real time epistemic uncertainty estimation is the key in safety in autonomous driving.
- Note that you can have extremely low model uncertainty while having high sensing uncertainty. But low epistemic uncertainty does not mean your model is good. Model uncertainty and model performance are two different dimension to characterize a model. 
- Model uncertainty is the uncertainty over the model's weights, and thus by definition cannot be evaluated using a single forward pass. --> need more investigation for real-time estimation.
- Out-of-data examples, which can be identified with epistemic uncertainty, cannot be identified with aleatoric uncertainty alone.
- Dropout variational inference is a practical way to estimate epistemic uncertainty.
- Aleatoric uncertainty can be directly modeled as the second output of the model together with yhat, using a single forward pass and thus enables real-time application. This second output aleatoric uncertainty can be interpreted as **learned loss attenuation**. **We do not need "uncertainty label" to learn uncertainty.** Rather we only need to supervise the learning of the regression task. We learn the variance $\sigma^2$ implicitly from the loss function.

#### Technical details
- Regression of a constant in the denominator is numerically unstable. Usually we do re-parameterization to avoid that.
- Normalized score vector obtained after sigmoid do not necessarily capture model uncertainty. It needs to be calibrated. 
- Epistemic uncertainty decreases as the training dataset gets larger. Aleatoric uncertainty does not increase for out-of-data examples, where epistemic certainly does.
- The loss
	- Regression
	$$L=\frac{1}{D}\sum_i \frac{1}{2} exp(-s_i) ||y_i - \hat{y}_i|| + \frac{1}{2} s_i$$
	- Classification
	$$L=\sum_i log\frac{1}{T}\sum_t exp(\hat{x}_{itc} - log\sum_{c'} exp \hat{x}_{itc'})$$
	$$\hat{x}_{it} = f_i + \sigma_i \epsilon_t, \epsilon_t \sim N(0, I)$$
- The uncertainty in prediction
	- Regression
	$$Var(y) = \frac{1}{T} \sum_t \hat{y}_t^2 - (\frac{1}{T} \sum_t \hat{y}_t)^2 + \frac{1}{T} \sum_t \sigma^2_t$$
	- Classification
	$$Var(y) = H(p) = -\sum p_c \log(p_c)$$
	$$p = \text{Softmax}(\hat{x})$$
- In summary, aleatoric uncertainty is predicted directly from the input image as the additional model output, but the epistemic uncertainty can only be modeled by variational inference. 
- Removing pixels with higher uncertainty will increase precision.

#### Notes
- [Alex Kendall's blog](https://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/)
- [Alex Kendall's slides](https://alexgkendall.com/media/presentations/oxford_seminar.pdf)
- [Uncertainty in Deep Learning](https://www.youtube.com/watch?v=HRfDiqgh6CE) (youtube video at PyData Tel Aviv)
- [Wikipedia entry](https://en.wikipedia.org/wiki/Uncertainty_quantification#Aleatoric_and_epistemic_uncertainty)
- **Real-time epistemic uncertainty** is the future direction of research. Can we use a neural network to learn this?
- Etymology: *Episteme*: knowledge in Greek. *Aleator*: dice player in Latin.