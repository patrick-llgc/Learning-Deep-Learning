# [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)

_May 2020_

tl;dr: Improvement over SENet.

#### Overall impression
Channel attention module is very much like SENet but more concise. Spatial attention module concatenates mean pooling and max pooling across channels and blends them together. 

Each attention is then used sequentially with each feature map.  

![](https://vitalab.github.io/article/images/convblock/fig2.png)
![](https://vitalab.github.io/article/images/convblock/fig1.png)

The Spatial attention module is modified in [Yolov4](yolov4.md) to a point wise operation.

#### Key ideas
- Summaries of the key ideas

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

