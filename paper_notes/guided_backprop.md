# [Guided backprop: Striving for Simplicity: The All Convolutional Net](https://arxiv.org/pdf/1412.6806.pdf)

_September 2019_

tl;dr: Guided prop for visualizing CNN efficiently. Also, max pooling can be replaced by conv with larger strides.

#### Overall impression
Backprop visualizes contribution of pixels to a classification results via backprop, but mask out the negative gradient. This leads to less noise in the visualized saliency map as compared to vanilla backprop.

The idea is summarized well in this [blog post](https://towardsdatascience.com/feature-visualisation-in-pytorch-saliency-maps-a3f99d08f78a) by the author of [FlashTorch](https://github.com/MisaOgura/flashtorch).

The idea can be combined with class activation map (CAM) or grad-CAM as well. But as shown in [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations), the difference between guided backprop (GB) and grad-CAM is not that big.

#### Key ideas
- Summaries of the key ideas

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

