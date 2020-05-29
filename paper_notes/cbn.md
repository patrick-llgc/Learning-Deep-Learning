# [CBN: Cross-Iteration Batch Normalization](https://arxiv.org/abs/2002.05712)

_May 2020_

tl;dr: Improve batch normalization when minibatch size is small.

#### Overall impression
Similar to [GroupNorm](groupnorm.md) in improving performance when batch size is small. It accumulates stats over mini-batches. However, as weights are changing in each iteration, the statistics collected under those weights may become inaccurate under the new weight. A naive average will be wrong. Fortunately, weights change gradually. In Cross-Iteration Batch Normalization (CBM), it estimates those statistics from k previous iterations with the adjustment below.

![](https://miro.medium.com/max/1400/1*7iIrwiilfm-V1S07eAhq9A.jpeg)

#### Key ideas
- Summaries of the key ideas

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

