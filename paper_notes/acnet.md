# [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](https://arxiv.org/abs/1908.03930)

_January 2021_

tl;dr: Train with 3x3, 3x1 and 1x3, but deploy with fused 3x3.

#### Overall impression
This paper take the idea of BN fusion during inference to a new level, by fusing conv kernels. It has **no additional hyperparameters during training, and no additional parameters during inference, thanks to the fact that additivity holds for convolution.**

It directly inspired [RepVGG](repvgg.md), a follow-up work by the same authors. 

#### Key ideas
- Asymmetric convolution block (ACB)
	- During training, replace every 3x3 by 3 parallel branches, 3x3, 3x1 and 1x3. 
	- During inference, merge the 3 branches into 1, through BN fusion and branch fusion.

![ACNet](https://pic3.zhimg.com/80/v2-c530c6327fbc39319f6c44eca3291e12_1440w.jpg)
- ACNet strengthens the skeleton
	- Skeletons are more important than corners. Removing corners causes less harm than skeletons. 
	- ACNet aggravates this imbalance
	- Adding ACB to edges cannot diminish the importance of other parts. Skeleton is still very important.


#### Technical details
- Breaking large kernels into asymmetric convolutional kernels can save computation and increase receptive field cheaply.
- ACNet can enhance the robustness toward rotational distortions. Train upright, and infer on rotated images. --> but the improvement in robustness is quite marginal.

#### Notes
- [Review on Zhihu](https://zhuanlan.zhihu.com/p/131282789)

