# [TensorMask: A Foundation for Dense Object Segmentation](https://arxiv.org/abs/1903.12174)

_April 2020_

tl;dr: Aligned channel2spatial boosts the performance of instance segmentation than direct channel2spatial.

#### Overall impression
The paper proposes a relatively rigorous formulation for 4D tensor that unifies DeepMask and InstanceFCN into one framework. The paper seems to be overly complicated to convey two simple ideas: We need to align channel2spatial, and we need large masks for large objects.

The key question to dense instance segmentation: why cannot we naively adopt [CenterNet](centernet.md) architecture for instance segmentation?

The answer is that training a neural network with $480^2$ channels is intractable. Thus a tradeoff has to be made for $H \times W \times C$. Either predicts a coarse mask and rely on bilinear upsampling and feature alignment to gain better masks, as in [TensorMask](tensormask), or predicts full resolution masks at coarse location grids such as [SOLO](solo.md).

#### Key ideas
- Each mask is a HxW tensor. Dense prediction would require a 4D tensor representation, HxWxHxW. First two dimension are at each physical location. The latter two dimensions are the mask dimensions.
- Two main ideas:
	- First, channel2spatial can have **Natural representation** (direct channel2spatial) or **Aligned representation** (aligned channel2spatial). The authors demonstrated that aligned representation is one key ingredient to achieve better performance for dense mask prediction.
	- Second, being able to predict **large masks** for large object (tensor bipyramid) boosts instance segmentation performance. <-- using constant resolution mask in MaskRCNN is one bottleneck for segmenting large objects. 

![](https://deeplearn.org/arxiv_files/1903.12174v1/x3.png)

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

