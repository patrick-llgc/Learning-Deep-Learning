# [Actions as Moving Points](https://arxiv.org/abs/2001.04608) 

_January 2020_

tl;dr: CenterNet for video object detection.

#### Overall impression
This extends [CenterNet](centernet.md) as Recurrent SSD extends SSD.

However it is still using box-based method to generate bbox and then link them to action tublets. This is more of a bottom up approach as compared to [recurrent ssd](recurrent_ssd.md).

Drawbacks and limitations: The main drawback is that it takes in K frames (K=7) frames at the same time. It is not suitable for fast online inference. It does support multiple object detection at the same time, same as CenterNet. 

#### Key ideas
- Summaries of the key ideas

#### Technical details
- Summary of technical details

#### Notes
- [code tbd](https://github.com/mcg2019/MOC-Detector)

