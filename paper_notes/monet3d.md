# [MoNet3D: Towards Accurate Monocular 3D Object Localization in Real Time](https://arxiv.org/abs/2006.16007)

_November 2020_

tl;dr: Encodes the local geometric consistency (spatial correlation of neighboring objects) into learning.

#### Overall impression
The idea is similar to enforcing certain order in prediction. It learns the second degree of information hidden in the GT labels. It incorporates prior knowledge of geometric locality as regularization in the training module.

The writing is actually quite bad with heavy use of non-standard terminology.

#### Key ideas
- Local similarity constraints as additional regularization. If two objects are similar (close-by) in GT, then they should be similar in prediction as well.

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

