# [An intriguing failing of convolutional neural networks and the CoordConv solution](https://arxiv.org/abs/1807.03247)

_August 2019_

tl;dr: Predicting coordinate transformation (predicting x and y directly from image and vice versa) with Conv Nets are hard. Adding a mesh grid to input image helps this task significantly. 

#### Overall impression
The paper results are very convincing, and the technique is super efficient. Essentially it only concats two channel meshgrid to the original input.

[RoI10D](roi10d.md) cited this paper. 

#### Key ideas
- Other coordinates works as well, such as radius and theta.
- The idea can be useful for other tasks such as object detection, GAN, DRL, but not so much for classification.

#### Technical details
- Summary of technical details

#### Notes
- Uber made a [video presenting this paper](https://www.youtube.com/watch?v=8yFQc6elePA).
- A concurrent paper from VGG has more theoretical analysis [Semi-convolutional Operators for Instance Segmentation](https://arxiv.org/abs/1807.10712) <kbd>ECCV 2018</kbd>.