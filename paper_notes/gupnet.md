# [GUPNet: Geometry Uncertainty Projection Network for Monocular 3D Object Detection](https://arxiv.org/abs/2107.13774)

_August 2021_

tl;dr: Uncertainty prediction of 3D height transfer to uncertainty of depth.

#### Overall impression
The multi-task learning part is quite interesting, but the depth prediction part lacks clarity and insight (it is more like a post-hoc experiment report).

The relationship between height and depth indeed can be mined, but with a sutble difference that the projected $h_{3d}$ does not necessarily match the height of the 2D bbox $h_{2d}$.

#### Key ideas
- Height-guideed depth prediction
	- Uncertainty prediction of 3D height. This can be transferred to uncertainty of depth (H_3d is proportional to d given accurate prediction of H_2d and a fixed focal length). --> The height guided depth prediction is similar to [MonoFlex](monoflex.md) and [GS3D](gs3d.md). 
	- No direct prediction of depth but rather only a bias term. --> This bias term is actually **critical** as the height of the 2d bbox $h_{2d}$ is not the same as the reprojected $h_{3d}$. Alternatively the network can predict a reprojected H_3d such as [RTM3D](rtm3d.md).
	- Verified in the ablation study that if we just use height to transfer to depth (GeP), the accuracy actually drops. Thus we have to add that bias term (GeU).
	- In [GS3D](gs3d.md), this bias is estimated to be 93% of the bbox height.
- UnC: the regressed depth uncertainty $\sigma_d$ can be mapped to a value [0, 1] and used to predict 3D confidence during inference $p_{depth} = \exp(-\sigma_d)$, and $p_{3d} = p_{2d} p_{depth}$
- HTL (hierarchical task learning) strategy
	- HTL is inspired by the motivation that each task should start training after its pre-tasks have been trained well. HTL ensures only when a task is learned good enough then start training of another task.
	- It solves the instability of the initial training phase where the estimation of 2D/3D heights will be noisy, leading to bad depth prediction.
	- DF(t) computes last K epochs to reflect the mean change trend. --> Not iterations! Iterations would be too noisy.
	- ls (learning situation) score = 1 - DF(t)/DF(K). This means if the loss change trend is similar to the first K epochs then ls score is small. 
	- A task will start training only when all pretasks achieve high ls.

#### Technical details
- Glossary in this paper is a bit confusing, and here is what I think they mean.
	- GeP: reprojected height to infer depth directly
	- GeU: GeP + depth bias
	- UnC: use GeU uncertainty to guide inference
- Architecture: Used two-stage detection and used [CoordConv](coordconv.md) to enhance the feature map. --> If only local yaw is predicted, this is actually not needed.
- Laplacian prior is assumed to be able to use L1 loss as a baseline in the uncertainty loss. Gaussian prior should be used to use L2 loss as a baseline.

#### Notes
- Questions and notes on how to improve/revise the current work  

