# [Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding](https://arxiv.org/pdf/1511.02680.pdf)

_June 2019_

tl;dr: Estimate the variance of segmentation uncertainty with dropout inference samples. Use the mean for prediction. The idea is quite similar to TTA (test time augmentation).

#### Overall impression
The paper provides a practical way to evaluate the uncertainty (this is the epistemic uncertainty), at a cost at inference time.  Refer to [Bayesian DL](uncertainty_bdl.md) for integration with aleatoric uncertainty. 

#### Key ideas
- Sampling with dropout performs better than weight averaging (normal dropout behavior during eval). Sampling dropout performs better than weight averaging after approximation with 6 samples. The performance saturates with 40 samples. 
	- This comes at inference time cost, but is naively parallelizable.
- The results also show that when the model predicts an incorrect label the model uncertainty is very high.
- Class boundaries usually display high level of uncertainty.
- Objects that are occluded or at a distance from the camera are are uncertain.
- The uncertainty score is inversely proportional to occurrence and accuracy. The model is more confident about classes which are easier and occur more often.
- The accuracy improves when we use a tighter threshold to filter out non-confident results. Uncertainty is an effective measure of accuracy.

#### Technical details
- **No need to use dropout layer after every layer.** Get the optimal architecture first by test placing dropout in different places. Then keep using dropout during inference (variational inference).

#### Notes
- [poster](https://alexgkendall.com/media/presentations/bmvc17_bayesian_segnet_poster.pdf)
