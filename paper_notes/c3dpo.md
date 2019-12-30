# [C3DPO: Canonical 3D Pose Networks for Non-Rigid Structure From Motion](https://arxiv.org/abs/1909.02533)

_December 2019_

tl;dr: Infer 3D pose for non-rigid objects by introducing DL to non-rigid structure-from-motion (NR-SFM).

#### Overall impression
C3DPO transforms closed-formed matrix decomposition problem into a DL-based parameter estimation problem. This method is faster and also can embody prior info that is not apparent in the linear model. 

A challenge in NR-SFM is the ambiguity of internal object deformation (or pose in this paper, non-rigid motion) and viewpoint changes (rigid motion). C3DPO introduces a canonicalization network to encourage the consistent decomposition. 


#### Key ideas
- The main takeaway from this work: Work as many constraints as possible into loss. Use any mathematical cycle-consistency to constrain learning.
- Use deep learning to supplement maths, not to replace math. 

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

