# [DC: Depth Coefficients for Depth Completion](https://arxiv.org/abs/1903.05421) 

_December 2019_

tl;dr: Encode depth in a simplified one-hot encoding (DC) and cross entropy loss reduces over-smoothing in depth estimation.

#### Overall impression
Similar to the idea of [SMWA](smwa.md) to address the "long tail" problem. This problem can be also referred to as edge bleeding, over-smoothing, or mixed depth. It features spurious depth estimation in mid-air and connecting surfaces between separate objects. 

> [DC](depth_coeff.md) focuses on depth completion while [SMWA](smwa.md) focuses on depth estimation from stereo pairs.

It also acknowledges that the problem is a multi-modal problem and using L1 or L2 leads to spurious estimation in-between modes. 

The idea of using an N-channel but 3-hot depth encoding is similar to the soft one-hot encoding used in [SMWA](smwa.md). In SMWA it also uses cross entropy for regression. DC gives a better explanation why cross entropy is a better loss than L1 or L2.

The input and loss modification is based on [sparse-to-dense](sparse_to_dense.md) and is easy to implement.

#### Key ideas
- One-hot encoding of depth and the use of cross-entropy loss solves the problem of mixed-depth problem.
- **RMSE favors over-smoothed depth estimation and thus is not a reliable metric.**

#### Technical details
- Depth reconstruction: either weighted average, or pick the single modal weighted average (eq 7). --> However the paper did not go to details on this.
- **The output dense depth leads to improved lidar performance**. --> this is to be compared with [pseudo lidar e2e](pseudo_lidar_e2e.md) which suffers from long tail problem.

#### Notes
- But after thinking about this again: how does changing the one-hot encoding into soft one-hot encoding help in alleviating the problem? How does cross entropy come to rescue when N degenerates to 1? Then it becomes softmax loss. --> cross entropy enables multi-modal?

