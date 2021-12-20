# [BN_FFN_BN: Leveraging Batch Normalization for Vision Transformers](https://openaccess.thecvf.com/content/ICCV2021W/NeurArch/papers/Yao_Leveraging_Batch_Normalization_for_Vision_Transformers_ICCVW_2021_paper.pdf) 

_December 2021_

tl;dr: Introducing BN into vision based transformers via a **BN-FFN-BN** structure.

#### Overall impression
Inherited from the NLP tasks, vision transformers take Layernorm (LN)as default normalization technique when applied to vision tasks. 

This is an alternative to [PowerNorm](powernorm.md) to replace LN in transformers.

#### Key ideas
- BN is faster in inference than LN due to avoidance of calculating the mean and variance statistics during inference.
- LN is suitable for NLP tasks as the input has variable length, and LN in NLP only calculates statistics in the channel dimension without involving batch or sequence length dimension.
- Naively replacing LN with BN leads to crashes. The crashes are due to un-normalized FFN blocks. Thus a BN layer is added in-between the two linear layers in each FFN block. This leads to stabilization of training and 20% faster inference.


#### Technical details
- Normalization methods: batch-related and batch-irrelevant. 
	- LN is best suited for NLP tasks
	- GN is good for small batch size and for dense prediction tasks
	- IN is good for style transfers
- For a 4D tensor input (NCHW)
	- BN has 2C elements of statistics, where each mean and var are computed across NHW in training. In inference, the averaged values are used without recalculating.
	- LN normalizes input along C leading to 2NHW statistics. It calculates statistics for each sample independently. This requires calculation in both training and evaluation.

#### Notes
- Questions and notes on how to improve/revise the current work
