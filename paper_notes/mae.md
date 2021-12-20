# [MAE: Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

_November 2021_

tl;dr: Scalable unsupervised pretraining of vision model by masked image modeling.

#### Overall impression
This paper is very enlightening.

This paper rushed the publication of other contemporary work such as [SimMIM](simmim.md) and [iBOT](ibot.md). The clarity of the message, the depth of insight, the craft of engineering consideration, the coverage of ablation study of MAE is significantly superior to the others.

#### Key ideas
- Masking a high proportions of the input image yields a nontrivial and meaningful self-supervisory task.
- Language and vision have very different information density. 
	- Languages are human generated signals, highly semantic and information dense.
- Asymmetric encoder and decoder
	- Encoder only 
	- Saves significant computation for transformer-based backbone
- Downstream tasks (object detection, instance and semantic segmentation) all surpassed supervised pretraining.

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  
