# [HDMapNet: An Online HD Map Construction and Evaluation Framework](https://arxiv.org/abs/2107.06307)

_July 2021_

tl;dr: Structured static BEVNet with optional lidar camera early fusion.

#### Overall impression
The paper focuses on prediction of vectorized map elements in the birds-eye-view for scene understanding. The perspective to BEV transformation is not done via IPM but rather with a MLP. The extrinsics are only used to piece the cameras together to the ego frame (thus with only translation, not rotation).

The static BEVNet is essentially an online map. learning framework that makes use of onboard sensors and computation to estimate local maps.

#### Key ideas
- View transformer: both neural feature transformation and geometric projection. HDMapNet does not predict depth of monocular cameras at all but use MLP to map pixels from perspective to BEV.
- Image encoder
	- perspective view image encoder: EfficientNet-B0 pretrained.
	- neural view transformer (MLP): from $H_{pv} \times W_{pv}$ perspective view feature map spatial dimension to $H_c \times W_c$ top down spatial dimension. The MLP is shared channel-wisely and does not change the feature dimension (Note: should be num of feature channels?). --> This proves to be better than [Lift Splat Shoot](lift_splat_shoot.md).
	- the camera features are then transformed to **ego vehicle frame** with using extrinsics. Final feature is the **average** of $N_m$ camera features.
- Point cloud encoder
	- point-pillars. 
	- concatenated with camera feature if lidar branch exists
- BEV decoder
	- semantic segmentation
	- instance embedding: push-pull loss
	- lane direction: 2-hot encoding. For background the label is 0.
- Postprocessing
	- vectorization: DBSCAN + NMS, then connecting with direction.
- Evaluation
	- Semantics metrics
		- Eulerian metric: pixel value, IoU.
		- Lagrangian metric: structured outputs, Chamfer distance
	- Instance metrics
		- TP, F1 score

![](https://cdn-images-1.medium.com/max/1600/1*HwMxIxdiuEewezEp7VSk_Q.png)

- vectorized output of 3 classes (lane line, boundary and pedestrian crossing)
- optional early fusion of camera and lidar

#### Technical details
- ICP and NDT (normal distribution transform) to do lidar alignment/registration.
- Drawbacks of IPM: IPM assumes a flat ground plane. Predictions in each view are not easily combined to a holistic view.
- MLP is better than IPM (assuming ground plane) and depth prediction (Lift Splat Shoot).
- Different categories are not equally recognizable in one modality. Road boundary are more recognizable with lidar than lane lines. 

#### Notes
- The BEV decoder can be useful for top-down view image recognition. This solves two issues
	- Clustering of close-by objects, even if they intersect or are close by
