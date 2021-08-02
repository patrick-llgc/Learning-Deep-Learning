# [FIERY: Future Instance Prediction in Bird's-Eye View from Surround Monocular Cameras](https://arxiv.org/abs/2104.10490)

_June 2021_

tl;dr: Prediction in BEV from monocular cameras. 

#### Overall impression
This paper is heavily inspired by [Lift Splat Shoot](lift_splat_shoot.md) in lifting multicamera features to 3D and then splat onto BEV view. However they are different too. 

- [Lift Splat Shoot](lift_splat_shoot.md) focuses on motion planning of ego car in the "shoot" part, while [FIERY](fiery.md) focuses on behavior prediction of other traffic participants.
- [FIERY](fiery.md) improves the semantic segmentation of [Lift Splat Shoot](lift_splat_shoot.md) to instance segmentation.
- [FIERY](fiery.md) also introduced the temporal component and leverages past frames.



#### Key ideas
- Center + semantic = instance segmentation + detection. Looks like Baidu's [CNN_seg](cnn_seg.md).
- Architecture
	- Lifting to 3D
		- Input: n=6 cameras. Image from camera k at time step t. $I_t^k$
		- Encode each image $e_t^k = E(I_t^k) \in R^{(C+D)\times H_e\times W_e}$
		- Outer product: $u_t^k = e_{t, C}^k \otimes e_{t, D}^k \in R^{C \times D \times H_e \times W_e}$
			- the depth probability act as a form of self-attention, modulating the features accoridng to which depth plane they are predicted to belong to.
	- Project to BEV
		- features are sum-pooled, $x_t \in R^{C\times H \times W}$
		- 0.5 m per pixel, 100m x 100m, same as [Lift Splat Shoot](lift_splat_shoot.md).
	- Temporal representation
		- Past features are transformed to present refernece frame using known past ego motion. --> localization with smooth DR pose should be good enough. Image $i \in {1, ... t-1}$ warped to t (present time) $x_i^t$ .
		- Concatenated and feed into a temporal module $s_t = T(x_1^t, ..., x_t^t)$ with $x_t^t = x_t$. T is a 3D conv networks.
		- 1 seconds in the past, to predict 2 seconds in future. In NuScenes dataset, 1+2 --> 4 frames @ 2Hz; In Lyft dataset, 1+5 --> 10 frames @ 5Hz.
	- Present and future distribution
		- TBD

#### Technical details
- The BEV backbone of combining multiple cameras has the functionality of sensor fusion. Instead of [Lift Splat Shoot](lift_splat_shoot.md) that does wholistic motion planning directly, [FIERY](fiery.md) actually does the prediction first, and the authors mentioned that they will work on the planning part later. It is a bit like [MP3](https://arxiv.org/abs/2101.06806) by Uber ATG.
- Non-parametric (what?) future trajectories.
- [Social LSTM](social_lstm.md) is actually done from a surveillance view point (between perspective onboard cameras and BEV).
- VectorNet and coverNet are good SOTA papers as of 2021 for prediction.

#### Notes
- Code available at [github](https://github.com/wayveai/fiery). This introduces the temporal module that is quite interesting. 
- [Wayve's blog review](https://wayve.ai/blog/fiery-future-instance-prediction-birds-eye-view/)