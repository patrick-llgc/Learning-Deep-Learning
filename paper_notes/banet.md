# [BA-Net: Dense Bundle Adjustment Networks](https://arxiv.org/abs/1806.04807)

_July 2020_

tl;dr: Feature-metric differentiable bundle adjustment.

#### Overall impression
BA-Net proposed to do **BA on feature maps** to avoid sensitivity to photometric changes in the input, while still leverages denser information than keypoints. It also uses a new formulation of depth prediction from a series of prototype depth maps (cf [YOLACT](yolact.md)). --> The compact depth representation by coefficient is very similar to CodeSLAM where an AutoEncoder is used to compress depth maps. This compact depth representation decreases the search space of LM algorithm.

Note that there is no PoseNet to predict ego motion. The output of the BA layer is the camera pose sequence and point cloud depths. 

The performance on KITTI looks too good to be true. Almost on par with supervised method [DORN](dorn.md). The sqr rel has some issues per author's answer here in [知乎](https://www.zhihu.com/question/306551694/answer/575851635).


##### Background of BA
- Keypoint BA
	- Only information from corners and blobs
	- Feature matching with RANSAC still gives outliers
- Photometric BA: Direct method uses dense pixel values and leverages more information
	- Sensitive to initialization
	- Sensitive to photometric calibration (camera exposure, white balance)
	- Sensitive to outliers such as moving objects
- Feature-metric BA:
	- Higher level feature, insensitive to photometric calibration and MOD outliers
	- Less sensitive to initialization as it has more clear global minimum
	- It is like doing DSO with learned and more robust feature map (rather than original image).


#### Key ideas
- Bundle Adjustment is hard to integrate into neural network due to two reasons.
	- Iterative: this can be 
	- If/Else switch on predicting the damping factor.
- BA-Net fixed the iteration to 5, and uses an MLP to predict the damping factor. This makes BA end-to-end differentiable.
- Architecture
	- Input: from previous $X$ (state: camera pose and point depth) and Feature maps, error is computed. Error is fed into an MLP to get damping factor. 
	- Damping factor and Diagonal matrix from Jacobian yields $\Delta X$
	- $X$ gets updated and feed into next iteration.
- Basis depth map
	- instead of dense prediction, BA-Net predicts 128 basis depth channels and predict 128-dim weighting factor w. 

#### Technical details
- Uses 5 consecutive frames for optimization on KITTI and achieves 

#### Notes
- Questions and notes on how to improve/revise the current work  

