# [D4LCN: Learning Depth-Guided Convolutions for Monocular 3D Object Detection](https://arxiv.org/abs/1912.04799)

_June 2020_

tl;dr: Use depth map to generate dynamic filters for depth estimation.

#### Overall impression
The idea of depth aware convolution and the idea of the 2D and 3D anchor both come from [M3D RPN](m3d_rpn.md). 

#### Key ideas
- 2D/3D anchors. Average the 3D shape that are associated with the 2D bbox. This forms the 3D anchor. This operation allows the neural network only focus on the residual adjustment from the anchors and significantly reduces the task difficulty.
- Filter generation network generates **dynamic local filters**, using depth map as input. 
	- Generate a filter volume the same size as the feature map
	- Shift filter volumn by 3x3 grid and average, this approximates a local filtering operation.
	- Each location also learns a different dilation rate.
- Loss function:
	- Multiple loss terms weighted by focal loss style $(1-s_t)^\gamma$, where $s_t$ is the cls score of target class. 
	- For each anchor, there are 4 (bbox) + 2 (proj 3D center) + 3 (whl) + 1 (depth) + 1 (yaw) + 3x8 (8 corners) + n_c (cls, =4) = 35 + 4 = 39 output bits. --> this formulation is similar to [M3D RPN](m3d_rpn.md) and [SS3D](ss3d.md).
	- corner loss helps.

#### Technical details
- Pseudo-lidar discards semantic information. This may be addressed by [PatchNet](patchnet.md).

#### Notes
- [Github repo](https://github.com/dingmyu/D4LCN)
- Trident network utilizes manually defined multi-head detectors for 2D detection.

