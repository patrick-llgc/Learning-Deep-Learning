# [SKNet: Selective Kernel Networks](https://arxiv.org/abs/1903.06586)

_May 2020_

tl;dr: A new plug-and-play module of SKNet. Two-stream SENet. 

#### Overall impression
This is a solid work to extend SENet (squeeze and excitation). It chooses adaptive receptive field. Either 3x3 or 5x5. 

Compared to inception network, which has multiple parallel path of diff RF, SKNet adaptively chooses which path to focus more.

This inspired [ResNeSt](resnest.md) and is actually almost exactly the same. 

#### Key ideas
- Split, Fuse and Select. 
![](https://raw.githubusercontent.com/implus/SKNet/master/figures/sknet.jpg)
- SK Unit is a plug-and-play module and used to replace a normal 3x3 conv. 

#### Technical details
- 5x5 --> 3x3 with dilation = 2.

#### Notes
- Questions and notes on how to improve/revise the current work  

