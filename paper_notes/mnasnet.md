# [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/pdf/1807.11626.pdf)

_May 2019_

tl;dr: Search the neighborhood of MobileNetV2.

#### Overall impression
One of the main challenge of NAS is its vast search space. This paper uses [MobilenetsV2](mobilenets_v2.md) as a starting point and significantly reduces the search space. 

The algorithm can be seen as an evolution algorithm, just a glorified for loop.

#### Key ideas
- Combine model Accuracy and latency into a cost function $ACC(model) \times [LAT(model)/TAR]^w$, with w ranging from -0.07 (in MobilenetsV3 Large, or MNasNet-A1) to -0.15 (in MobilenetsV3 Small). -0.07 was obtained by observing that empirically model accuracy improves by 5% when doubling latency.

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

