# [Double Descent: Reconciling modern machine learning practice and the bias-variance trade-of](https://arxiv.org/abs/1812.11118)

_January 2020_

tl;dr: Machine learning models will generalize better once going beyond the interpolation peak. 

#### Overall impression
This is a mind-blowing paper. It extends the U-shaped bias-variance trade-off curve to a double descent curve. Beyond the interpolation threshold where the model starts to have zero empirical/training risk, test risk starts to dropping as well.

![](https://preview.redd.it/g4q983jk7lq21.png?width=1029&format=png&auto=webp&s=4d5d6498b6f48defbe4606576f99b2cd772ba863)

#### Key ideas
- Background
	- Conventional wisdom in ML is concerned in finding the sweet spot between underfitting and overfitting.
	- Modern DL architectures overfits to training data well (overfit with zero training error, or interpolation). Even with some corrupted labels the DL models can generalize well.
	- Historically this has been overlooked as most models are relatively small. Regularization of any sort can change effective capacity of function class, thus prevent interpolating (exact fitting) and masking the interpolation peak. **Early stopping** also prevents practitioners to observe the long tail beyond interpolation threshold. 
- The double descent behavior exhibits in a wide range of models and datasets.
- The double descent for NN happen within a narrow range of parameters and will lead to observations that increasing size of network improves performance. 

#### Technical details
- SGD is one example of ERM (empirical risk minimization)

#### Notes
- [ICML 2019 Talk](https://www.facebook.com/icml.imls/videos/international-conference-on-machine-learning-live-grand-ballroom-a/2543954589165286/)
- [Mikhail Belkin's talk at Institute for Advanced Study](https://www.youtube.com/watch?v=5-Kqb80h9rk)
- Andrej Karpathy's blog [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)


