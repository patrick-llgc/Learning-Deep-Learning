# [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)

_Mar 2019_

tl;dr: Factorize normal 2D convolution operations into depth separable convolutions (depthwise convolution and pointwise convolution) to reduce latency as well as model size.

#### Overall impression
The way normal 2D conv op handles channel information is almost in a fully connected fashion. Each channel in the input is filtered and weighted into the output by different weights in different and corresponding layers in the conv kernel. Depthwise separable conv applies the same 2D conv kernel to all depths, and uses a pointwise conv (1x1 conv) to combine it. 

#### Key ideas
- Two ways to build small and efficient networks: compressing pretrained networks (quantization, hashing, pruning or distillation, or low-bit network); training small networks direclty with new architectures such as mobilenets, mobilenets v2 shufflenet, xception, etc.
- MobileNet applies a single filter to each input channel. 
- Computation cost:
	- Input: F x F x M (M channel)
	- Normal conv: F x F x M x N x K x K
	- Depthwise conv: F x F x M x K x K
	- Pointwise conv: M x N x F x F
	- Reduction of computation: 1/N + 1/K^2 ~ 1/K^2 ~ 1/9 (for 3x3 conv kernels)
	- N is in the order of 100 to 1000
	- MobileNets' 95% of computation is in 1x1 pointwise conv and can be implemented very efficiently.
- With and resolution multiplier: $\alpha$ and $\rho$ to control input and output channel numbers, and the input resolution.
- There is a log=linear dependence between accuracy and computation.

#### Technical details
- MobileNets show that it is more benefit to make the network thinner than shallower. 
- SqueezeNet [?] uses fewer parameters but more calculations. In this sense it is like DenseNet?

#### Notes
- TODO: Need to read mobilenet v2 and shufflenet, and xception.
- Comparison of mobilenet and xception. ([Kerasの作者@fcholletさんのCVPR'17論文XceptionとGoogleのMobileNets論文を読んだ](https://qiita.com/yu4u/items/34cd33b944d8bdca142d))
- Q: Isn't depth separable 2D conv the same as a 3D convolution with z=1? I need to write down a more detailed note about this.
- Geolocalization with PlaNet [?].
- Distillation works by training the classifier to emulate the outputs of a larger model instead of the GT labels. Could this be used to combat noisy label problems?