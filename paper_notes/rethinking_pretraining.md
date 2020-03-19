# [Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883)

_March 2020_

tl;dr: ImageNet pretraining speeds up training but not necessarily increases accuracy. 

#### Overall impression
Tons of ablation study. Another solid work from FAIR. 

We should start exploring group normalization 

#### Key ideas
- ImageNet pretraining does not necessarily improve performance, unless it is below **10k COCO images (7 objects per image. For PASCAL images where 2 objects per iamge, we see overfitting even for 15k)**. ImageNet pretraining does not gives better regularization and not help reducing overfitting.
- ImageNet pretraining is still useful in reducing research cycles.

#### Technical details
- GroupNorm with batch size of 2 x 8 GPUs.

#### Notes
- Questions and notes on how to improve/revise the current work  

