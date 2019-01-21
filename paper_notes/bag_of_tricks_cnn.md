# [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf)

_Jan 2019_

tl;dr: A list of tricks to improve the training of CNNs for image classificaiton.

#### Overall impression
The paper summarizes the CNN training tricks scattered in the papers published in the past few years. However the ablation test on some tricks show mixed results and is not conclusive. **Cosine decay** seems to be the most solid trick.

#### Key ideas
- Improvement on image classification accuracy generally lead to better transfer learning performance in object detection, but not necessarily on semantic segmentation.
- Model tweaks of ResNet
  - The initial 1x1 conv, stride 2 **throws away** 3/4 information in the feature map. Delay the stride 2 to 3x3 conv layer or use a 2x2 pooling first to downsize then perform 1x1 conv.
  - Replace 7x7, stride 2 conv with 3 3x3 conv. The computational cost increases quadratically with kernel size, so 7x7 is 5.4 times more expensive than 3x3.
- Training refinement
  - **Cosine decay** schedule (optionally with warmup) generally improves the training speed and final performance.
  - **Label smoothing**. Cross entropy loss encourages the output scores dramatically distinctive which potentially leads to overfitting. 
    \[
    q_i = 
      \begin{cases} 
       1-\epsilon & \text{if } i=y \text{ (1 class)} \\
       \epsilon/(K-1)       & \text{otherwise (K-1 classes)} 
      \end{cases}
    \]
    - Another way to curb overfitting with cross entropy is to clip gradient. 
  - Mixup training. Randomly sample two images and linearly blend images and labels. $ \hat{x}= \lambda x_i + (1-\lambda$) x_j$ for both image and labels. $\lambda$ is usually sampled in a Beta(a, a) distribution centered around 0.5.
- Large batch training may slow down the training process (see [Notes](#notes) section).
  - Linear scaling learning rate.
  - Learning rate warmup.
  - Zero $gamma$ in batch norm. Essenstially shortens the conv layers at the beginning.
  - No bias decay. Only L2 penalize the weight, not the bias to avoid overfitting.
  - Low precision training: switch from commonly used FP32 to FP16 leads to 2 to 3 times acceleration on V100. 
- Transfer learning
  - Object detection: generally speaking, better classifiation leads to better object detection.
  - Semantic segmentation: only cosine decay still improves semantic segmentation. All other tricks improves image classificaiton but decays semantic segmentaton. 

#### Technical details
- Old CNN architecture (ResNet50) with the tricks can outperform newer architecture (SE-ResNeXt-50). This shows the importance of parameter tuning.

#### Notes
- The comment that "The number of updates required to reach convergence usually increases with training set size" is confusing. There is a good debate [here](https://stats.stackexchange.com/questions/323570/convergence-of-stochastic-gradient-descent-as-a-function-of-training-set-size).

#### Additional Resouces
- This [blog](https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/) shows how to implement the tricks in keras.
