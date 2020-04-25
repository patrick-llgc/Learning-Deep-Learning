# [Mask Encoding for Single Shot Instance Segmentation](https://arxiv.org/abs/2003.11712) 

_April 2020_

tl;dr: Encode mask into lower dimensional representations for prediction.

#### Overall impression
The paper showed that directly predicting high dimensional masks 28x28 = 784 is tractable, but leads to worth performance than predicting a lower dimensional vector and recover mask from it (N=60). 

However this is still region based and pretty much depends on bbox. The binary masks are with respect to the bbox. 

The idea of doing PCA to compress shape is very similar to that of [RoI10D](roi10d.md). This is to be compared with [CondInst](condinst.md)

#### Key ideas
- Compressing 28x28 region based masks into lower dimensional vectors. Linear compression using PCA. 28x28=784 vector is projected to 60 dimensions with a projection matrix of size 784x60. 
- L2 loss on the encoded 60-d mask vector directly.
	- It works better than decoding the mask and perform mask pixel cls.
	- It also works better than directly predicting the high dim vector. It showed that it is possible to predict the 784 dimensional mask directly as a high dimensional vector. The performance is only 2 AP worse. <-- [SOLO](solo.md) can be designed even simpler, but perhaps with worse performance. 
- Minimize the reconstruction error:
	- $v = T u$
	- $\tilde{u} = Wv$
	- $T^*, W^* = \arg \min \sum |u - WTu|^2$


#### Technical details
- [TensorMask](tensormask.md) is very heavy in computation and inference. It uses 6x training schedule. [ExtremeNet](extremenet.md) usese even 8x training schedule. Normally 12 epoch is good enough. 
- [MEInst](meinst.md) 

#### Notes
- Questions and notes on how to improve/revise the current work  

