# [Delving into the devils of bird's eye view perception]()

_October 2022_

tl;dr: A bag of tricks for BEV perception.

#### Overall impression
The review did a good job summarizing the recent progress in BEV perception, including both vision, lidar and fusion methods, covering both academia and industry, and also covers a wide range of useful tricks. This can also be cross-referenced with the leaderboard version of BEVFormer [BEVFormer++](bevformer++.md).

#### Key ideas
- The methods of BEV perception can be divided into two categories, depending on the BEV transformation method
	- 2D-to-3D: reconstruction method
	- 3D-to-2D: use 3D prior to sample 2D images

#### Technical details
- Use the evolution algorithm or annealing algorithm in NNI toolkit for parameter tuning with a evaluation dataset.
- TTA tricks for competition and autolabel
	- WBF method

#### Notes
- Need to write a review blog on DETR and improved version (anchor-DETR, conditional DETR, DAB-DETR, DN_DETR, DINO, etc).
