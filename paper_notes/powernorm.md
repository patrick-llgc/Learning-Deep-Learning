# [PowerNorm: Rethinking Batch Normalization in Transformers](https://arxiv.org/abs/2003.07845)

_December 2021_

tl;dr: A better alternative to LayerNorm in transformers (even for NLP).

#### Overall impression
PowerNorm significantly outperforms LayerNorm in NLP, and works for CV tasks as well. It keeps the advantage of BN (fusion into subsequent layers) during inference. 

A similar work is done in [BN-FFN-BN](bn_ffn_bn.md) yet with a different focus. 

- [PowerNorm](powernorm.md) has a broader scope. It focuses on debugging why BatchNorm does not work for NLP and finding an alternative to LayerNorm. It mainly focuses on NLP task, and did not go into CV field, although this trick should work for vision transformers as well.
- [BN-FFN-BN](bn_ffn_bn.md) focuses on a very specific scope of why BatchNorm does not work for Vision transformers. It introduces one additional BN layer between two linear layers in one FFN, solving the training instability.

#### Key ideas
- Naive use of BN leads to significant performance drop for NLP tasks. This is due to that statistics of NLP data across the batch dimension exhibit large fluctuations throughout training.
- BN vs PN
	- BN uses the batch avg and variance for training. BN uses running stats during training for inference.
	- PN uses the **running** square avg for training. PN uses running stats during training for inference.

#### Technical details
- BatchNorm and LayerNorm sums across perpendicular dimensions. BatchNorm sums across NWH, and LayerNorm sums across C.
- BN helps training as it results in a smoother loss landscape.

#### Notes
- Questions and notes on how to improve/revise the current work
