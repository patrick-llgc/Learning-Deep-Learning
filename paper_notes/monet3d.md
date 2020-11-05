# [MoNet3D: Towards Accurate Monocular 3D Object Localization in Real Time](https://arxiv.org/abs/2006.16007)

_November 2020_

tl;dr: Encodes the local geometric consistency (spatial correlation of neighboring objects) into learning.

#### Overall impression
The idea is similar to enforcing certain order in prediction. It learns the second degree of information hidden in the GT labels. It incorporates prior knowledge of geometric locality as regularization in the training module. The mining of pair-wise relationship if similar to [MonoPair](monopair.md).

The writing is actually quite bad with heavy use of non-standard terminology. No ablation study on the effect of this newly introduced regularization.

#### Key ideas
- Local similarity constraints as additional regularization. If two objects are similar (close-by) in GT, then they should be similar in prediction as well. 
- The similarity is defined as $s_{ij} = \exp (-\Delta u_{ij}^2 - \Delta z_{ij}^2/\lambda)$
- The difference between the output for different vehicles are penalized according to this metric.

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

