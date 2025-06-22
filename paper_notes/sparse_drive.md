# [SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation](https://arxiv.org/abs/2405.19620)

_June 2025_

tl;dr: E2E driving with sparse representation.

#### Overall impression
Roughly along the same line of UniAD. Main contirbution of the paper is threefold.

* A fully sparse scene representation (instead of a computationally expensive dense BEV grid)
* Joint prediction and planning (UniAD is predict-then-plan)
* CAR (collision aware rescoring): reject scene prediction if jointly predicted prediction and planning have collision. 

#### Key ideas

#### Technical details
- CAR (collision aware rescoring)
	- CAR improves collision score significantly without regressing L2 error metrics. 
	- UniAD uses a post-optimization collision check against perception results. This turns out to be hurting driving behavior. 

#### Notes
- Questions and notes on how to improve the current work

