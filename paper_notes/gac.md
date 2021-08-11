# [GAC: Ground-aware Monocular 3D Object Detection for Autonomous Driving](https://arxiv.org/abs/2102.00690)

_August 2021_

tl;dr: Anchor-based method with a ground aware convolution module.

#### Overall impression
This paper is directly inspired by [M3D-RPN](m3d_rpn.md). It still uses anchors instead of anchor-free, and uses the postprocessing module to optimize yaw. 

#### Key ideas
- The key idea is a ground aware convolution (GAC) module. The network predicts the offers in the vertical direction and we sample the corresponding features and depth priors from the pixels below. 
- Depth priors are inspired by [CoordConv](coordconv.md) and are computed with perspective geometry with ground plane assumption.
![](https://cdn-images-1.medium.com/proxy/1*b31hiO4ynbDLRrXWEFF4aQ.png)

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

