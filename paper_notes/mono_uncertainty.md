# [MonoUncertainty: On the uncertainty of self-supervised monocular depth estimation](https://arxiv.org/abs/2005.06209)

_July 2020_

tl;dr: **Self-teaching** with pseudo-label is the best way for uncertainty estimation for monocular depth estimation.

#### Overall impression
The paper has a very good review session on monocular depth estimation. Tons of ablations studies. It evaluated 11 methods and combinations for predicting the uncertainty of self-supervised monocular depth estimation.

Verdict is: When pose is known, the difference between self-teaching and log-likelihood is minor. When pose is entangled in the loss function, self-teaching is much better to produce the uncertainty of depth.

#### Key ideas
- Uncertainty by image: **2 forward pass**
	- flipping: easiest one. 
- Empirical estimation: **N forward pass**
	- Dropout sampling: turn on random dropout during prediction
	- Bootstrapped Ensemble: same model, N diff initialization
	- Snapshot Ensemble: same model, N early stopped version with cyclic LR
- Predictive estimation: **1 forward pass**
	- Learned reprojection: anomaly prediction predicting the 
	- Log-likelihood Maximization: whatever loss + aleatoric uncertainty. --> [Learn SfM from SfM](learn_sfm_from_sfm.md)
	- **Self-teaching**: L1/L2 loss + aleatoric uncertainty
- Bayesian estimation
	- Combination of empirical and predictive estimation. Uncertainty is the sum of predicted uncertainty, plus the deviation of predicted depth from the averaged depth.
- The conclusions:
	- Bootstrap ensemble is about the same as snapshot ensemble, but slightly better.
	- For monocular setup, empirical methods does not work well. Self-teaching improves baseline while log-likelihood worsens baseline.
	- For uncertainty, self-teaching > log-likelihood > postprocessing.

#### Technical details
- Weakly supervised:
	- Noisy lidar depth
	- Model-based depth: SGM for stereo, SfM, and with their confidence
- Uncertainty metric:
	- Sparsification plot (see explanation in [Active learning in Lidar](deep_active_learning_lidar.md)) to compare with oracle sparsification. It is useful to compare each model with its oracle. It is usually normalized to 1 at 0 removal (max error).
	- Sparsification error is the difference between each model and its oracle sparsification. It is therefore possible to compare different models. It by definition starts at [0, 0].
	- The single metric summary of sparsification error is AUSE (area under the sparsification error)
	- The idea was first introduced in [Uncertainty Estimates and Multi-Hypotheses Networks for Optical Flow]() <kbd>ECCV 2018</kbd>

#### Notes
- Questions and notes on how to improve/revise the current work  

