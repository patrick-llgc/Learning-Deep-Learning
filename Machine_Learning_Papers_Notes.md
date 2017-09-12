# Machine Learning Papers Notes
### Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation ([link](https://arxiv.org/pdf/1609.08144.pdf)) 
- Three inherent weaknesses of Neural Machine Translation (that prohibited NMT to overtake phrase based machine translation):
	1. Slower training and inference speed;
		- To improve training time: GNMT is based on LSTM RNNs, which have 8 layers with residual connections between the layers to encourage gradient flow. To improve inference time, low-precision arithmetic are used, further accelerated by google's TPU.
	2. ineffectiveness in dealing with rare words;  
		- To effectively deal with rare words: sub-word units ("wordpieces") were used for inputs and outputs.
	3. failure to translate all words in the source.
		- To translate all of the provided input, a beam search technique and a coverage penalty are used.
- Phrase-based machine translation (PBMT), as a type pf statistical machine translation method, has dominated machine translation for decades. NMT has been used as part of the PBMT and achieve promising results, but end-to-end learning based on NMT for machine translation has only started to surpass PBMT recently.
	- attention mechanism, character decoder, character encoder, subword units have been proposed to deal with rare words.
- GNMT is a sequence-to-sequence learning framework with attention. In order to achieve high accuracy, GNMT has to have deep enough encoder and decoder to capture subtle irregularities in the source and target.
- TBC


### DeepEM3D: Approaching human-level performance on 3D anisotropic EM image segmentation [link](https://academic.oup.com/bioinformatics/article-abstract/33/16/2555/3096435/DeepEM3D-approaching-human-level-performance-on-3D?redirectedFrom=fulltext)


### Sensor fusion [link](https://www.youtube.com/watch?v=xDDN8Q0hJos)
- 2 approchaes to sensor fusion
	- fuse input data from sensors before analysis
	- fuse analysis output data
- Prerequisites of sensor fusion
	- sensor synchronization using GPS
	- Localization in 6D
		- GPS is not reliable or accurate in urban canyons


### U-net: Convolutional Networks for Biomedical Image Segmentation
- [Link](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- From classification to localization (a class label is supposed to be assigned to each pixel)
- Ciresan trained a network in a sliding window setup to predict the label of each pixel by providing an image patch surrdounding it. 
	- Pros: 
		- able to localize
		- training data is much larger than the number of training images
	- Cons:
		- inefficient due to redundancies
		- tradeoff between localization accuracy (small image patches and the less use of pooling layer) and the use of context (large image patches)
- U-net is based on **fully convolutional netwowrk**.
- Architecture
	- Contracting layers + expanding (upsampling) layers
	- Concatenation with the correspondingly cropped feature map from the contracting path
![](images/unet_arch.png)
- Data handling
	- Extrapolation by mirroring is used with valid padding on the boarder where data is missing.
	- Overlap-tile strategy is used to allow segmentation of arbitrarily large input.
	- Excessive data augmentation by applying **elastic deformation** for the network to learn such invariances. This is the key to train a segmentation network with very few (~30) annotated images.
	- Output is a series of map, each representing the probability of a pixel belonging to a certain class.
- Training
	- A large **momentum** (0.99) is used due to smaller batch size used (=1 image patch)
	- Deep neural networks usually has an objective with the form of a long shallow ravine leading to the optimum with steep walls on the sides. Standard SGD has very slow convergence rate after the initial steep gains. Momentum can be used to push the objective more quickly along the shallow ravine. [link](http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/) Therfore momentum update is almost always better than vanilla SGD. [CS231n](http://cs231n.github.io/neural-networks-3/#sgd)
	- Momentum is particularly important when using small batches because it allows derivatives to be integrated across batches. The smaller the batch size, the greater the momentum you may want to use. [link](http://tedlab.mit.edu/~dr/Lens/thumb.html)
- Evaluation
	- Review section in the summary of [ISBI-2012](http://journal.frontiersin.org/article/10.3389/fnana.2015.00142/full)
	- As segmentation algorithms are generally embedded in semiautomatic systems that enables human experts to correct the mistakes of the algorithms, it is useful to define a "nuisance metric", but it is highly subjective.
	- Human effort is required to correct **split errors** and **merge errors**, which can be used as proxies for the nuisance metric.
	- Pixrl error: the easiest measure of segmentation performance, but does not reflect the human effort involved to correct split or mergr error, thus inadequate.
	- **IOU** (IoU, intersection over union, or [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)) is a widely used pixel error metric to evaluate segmenation algorithms (~90% is pretty good)
	- **Rand error**: non-local, region based method. More robust and best matches qualitative human judgement.
		- Define $p_{ij}$ as the probability that a pixel belonging to segment i in S (predicted segmentation) and segment j in T (ground truth segmentation). The joint probability distribution satisfies $\sum_{ij} p_{ij} = 1$ by definittion.
		- $s_i = \sum_j p_{ij}$ is the probability of a randomly chosen pixel belonging to segment i in S.
		$$
		V^{Rand}_{split} = \frac{\sum_{ij}p_{ij}^2}{\sum_k t_k^2}, \quad\quad V^{Rand}_{merge} = \frac{\sum_{ij}p_{ij}^2}{\sum_k s_k^2}.
		$$
		- The merge score $V^{Rand}_{merge}$ is the probability that two randomly chosen voxels belong to the same segment in T, given that they belong to the same segment in S. The merge score is higher when there are fewer merge errors. The split score is defined similarly.
		- The Rand F-score is defined as the weighted harmonic mean
		$$
		V_\alpha^{Rand} = \frac {\sum_{ij} p^2_{ij}} 
		{\alpha \sum_k s_k^2 + (1-\alpha) \sum_k t_k^2}
		$$
		Generally $\alpha = 0.5$, which weighs split and merge errors equally. The Rand score is closely related to the Rand index.
		
		
### 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
- [Link](https://arxiv.org/abs/1606.06650)
- 3D U-net is an end-to-end training scheme for 3D (biomedical) image segmentation based on the 2D counterpart U-net. It also has the analysis (contracting) and synthesis (expanding) paths, connected with skip (shortcut) connections.
![](images/3dunet_arch.png)
- 3D U-net takes 3D volumes as input and process them with 3D operations (3D convolution, 3D max pooling, 3D up-convolution). 
- Biomedical images has the special advantage that properly applied rigid transformations and slight elastic deformations still yield **biologically plausible** images. In addition, each image already comprise repetitive structures with corresponding variations. Combined they allow efficient training of neural networks on sparsely annotated data. (In 3D, 2 volumes are enough to train a network from scratch and perform segmentation on the 3rd volume.)
- Batch normalization (BN) is used in preference to previously used He initialization (Gaussian distribution with $\sigma=\sqrt{2/N}$) for faster convergence.
- Weighted softmax loss function (0 for unlabled regions as they do not contribute to loss function, reduced weighting for frequently seen background, and increased weighting for rare classes, in the interest of more balanced influence from different classes on loss function)
- Results
	- BN: better accuracy (IoU) is achieved with BN
	- 3D context: contributes positively to the segmentation accuracy (compared with treating each slice independently).
	- Number of slices: diminishing return after a few slices (a couple of percentage of total number)


### V-Net: fully Convolutional Neural Network for Volumentric Medical Image Segmentation
- Objective function based on Dice coefficient
	- [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient): a statistic used for comparing the similarity of two samples
	- $S={\frac {2|X\cap Y|}{|X|+|Y|}}$, which is related to Jaccard index (IoU), $J=\frac{|X \cap Y|}{|X \cup Y|}$, in that $S = 2J/(1+J)$ and both $S, J \in (0, 1)$.

	
	
		
### Fully Convolutinal Networks for Semantic Segmentation (FCN)
- [link](https://arxiv.org/abs/1605.06211)
- FCN adapts the classification networks for dense prediction, making it capable of localizatio tasks as well. Both learning and inference are performed whole-image-at-a-time.
- Architecture
	- Typical classifier nets take fixed-sized inputs.
	- **KEY STEP**: Fully connected (FC) layers can also be viewed as convolutions with kernels that cover their entire input regions. Doing so cast the classification nets to fully convolutional networks (FCN) that take **input of any size** and make spatial output maps. 
	- Although resulting maps are equivalent to the evaluation of original net on particular input image patches, the computation is **highly amortized** over the overlapping regions of those patches.
	- Although casting the nets to a fully convolutional manner will provide some localization information, the resulting image will be one with lower resolution due to max-pooling layers.
	- Addition of **skip connections** helps with the coarse prediction limited by the stride (due to **max-pooling**) of the convolution layers. This combines *where* (localizatino information from shallower layers) and *what* (classification information from deeper layers) of the network.
- Skip connections:
![](images/fcn_arch.png)
	- FCN-32s: Start with VGG-16 and convolutionalize the fully connected layer, perform 32x upscaling from the final stride-32 layer.
	- FCN-16s: Start with FCN-32s, concatenate stride-16 with 2x upscaled stride32 layers first, then perform a 16x upscaling. 
	- FCN-8s: Start with FCN-8s, concatenate stride-8 with 2x upscaled stride-16 layers and 4x upscaled stride-8 layers.
	- The 2x interpolation layers is initialized to bilinear interpolation
- Training
	- **Fine-tuning** only the final classification layer only yields 73% of the the full fine-tuning performance. 
	- Training all-at-once yeilds similar results to training in stages (FCN-32 first, then FCN-16, finally FCN-8) yields very similar results but only takes about half the time. However each stream needs to be scaled by a fixed constant to avoid divergence.
	- Class balancing: by weight or sampling the loss. ^(c.f. 3D U-net, ==need more investigation==) Mildly unlablanced datasets (1:3, e.g.) do not need rebalancing.
	- **Upsampling** needs to be defined as convolution for end-to-end training and inference. See details in this [blog](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/).^(==need to read==)
- Evaluation
	- $n_{ij}$ is the number of pixels of class i predicted to be class j.
	- $t_i = \sum_j n_{ij}$ is the total number of pixels of class i.
	- pixel accuracy: $\sum_i n_{ii} / \sum_i t_i = \sum_i n_{ii} / \sum_{ij} n_{ij}$
	- mean accuracy: $(1/n_{cl}) \sum_i n_{ii}/t_i$
	- mean IU: $(1/n_{cl}) \sum_i n_{ii}/(t_i + \sum_j n_{ji} - n_{ii})$
- Momentum
	- Higher momentum is needed for batch size. $p^{1/k} = p'^{1/k'}$. For example, for momentun 0.9 and a batch size of 20, an equivalent training regime may be a momentum of $0.9^{1/20} \approx 0.99$ and a batch size of one, which is equivalent of **online learning**. In general, online learning yields better FCN models in less wall clock time.
	
- Extensions
	- [DIGITS 5 from Nvidia](https://devblogs.nvidia.com/parallelforall/image-segmentation-using-digits-5/)^(==need to read==)


## Blogs, Websites, slides, etc
- DeepLearningJP2016 [@SlideShares](https://www.slideshare.net/DeepLearningJP2016/presentations)

	
	
	
	
	
	
	
	