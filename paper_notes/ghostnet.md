# [GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907)

_February 2020_

tl;dr: A computationally efficient module to compress neural network by cheaply generate more feature maps

#### Overall impression
Instead of generating all n feature maps from all c channels in the input, generate m feature maps first (m < n), and then use cheap linear operation to generate n-m feature maps. The compression ratio is s. 

The linear operation is usually a 3x3 conv. Different from original 3x3 convolution which takes in all c channels in the input, the m-n are generated from each of the m features directly (injective 单摄).

The paper has a very good description of compact model design, including [mobilenets](mobilenets.md), [mobilenets v2](mobilenets_v2.md), [mobilenets v3](mobilenets_v3.md).

Model compression methods are usually bounded by the pretrained deep neural network taken as their baseline. The best way is to design such an efficient neural network that lends themselves to compression.

#### Key ideas
- Summaries of the key ideas

#### Technical details
- Summary of technical details

#### Notes
- [official github repo](https://link.zhihu.com/?target=https%3A//github.com/huawei-noah/ghostnet)

