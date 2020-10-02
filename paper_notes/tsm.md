# [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383)

_October 2020_

tl;dr: Plug-in TSM module to enable 2D conv nets for efficient video understanding. 

#### Overall impression
The idea seems to be inspired by [Shift as conv]() but naive implementation did not work, both in efficiency and performance. It uses efficient 2D conv operations and reaches good performance as compared to 3D conv.

It can be inserted into 2D CNNs to achieve temporal modeling at zero computation and zero parameters. Roughly same latency as 2D CNN baseline. The only cost is 1/4 of the feature maps of each residual block need to be saved. For ResNet-50, we only have to cache 0.9 MB data. TSM only uses 2D convolution which is highly optimized for hardwares.

TSM is orthoganal to [non-local](non_local_net.md), and boosts all 2D conv backbones. It leads to double digit improvement in some challenging datasets that has strong temporal relationship. 

It seems to be better than [convLSTM](https://arxiv.org/abs/1411.4389) <kbd>CVPR 2015 oral</kbd>.

#### Key ideas
- TSM shifts only 1/4 of the channels. 
	- Shifting all channels incurs high latency.
	- Shifting all channels harms spatial modeling of the network
- Residual shift keep a clean residual connection and avoids degraded spatial feature learning problem
- Multi-level temporal fusion. 
	- Conv-LSTM only does late fusion
- TSM online is slightly worse than TSM offline, but much better than baseline methods. 


#### Technical details
- High throughput: processes ~80 videos per second on a 16 GB Tesla P100 GPU.

#### Notes
- Questions and notes on how to improve/revise the current work  

