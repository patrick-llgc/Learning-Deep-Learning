# [BEV-Net: Assessing Social Distancing Compliance by Joint People Localization and Geometric Reasoning]()

_September 2021_

tl;dr: Metric BEV perception for pedestrians from surveillance cameras.

#### Overall impression
This paper builds upon crowd-counting methods. It detects head and foot of pedestrians and leverages a differentiable bilinear interpolation BEV transformation module to lift them to BEV.

#### Key ideas
- Head or foot?
	- Head is used in people counting and are move visible, but they are not on the same plane due to height variations. We cannot simply lifted to 3D with homography.
	- Foot is on the same ground plane but they are often occluded (~25% in cityUHK-X dataset)
	- Ablation study shows that we need both to achieve SOTA results.
- Architecture: Multi-branch Encoder and decoder
	- Pose (camera height and pitch) estimation with simple MSE error loss
	- Head keypoints prediction
	- Foot keypoints prediction
	- BEV-transform: feature level homography to lift head and foot keypoint feature maps to BEV with estimated pose.
- T_head (head feature map transformation) uses multi-bin with attention
	- As people's head is not on the same plane, a multi-bin approach is used to predict people's head in 1.1, 1.2, ..., 1.8 m, in total 8 height bins. 
	- The people are assigned to different height planes by self-attention mechanism.	The learned feature map is transformed to BEV and then fused with attention. 

	
#### Technical details


#### Notes
- BEV-transform assumes that the detected keypoints are on the same plane. This may work perfectly well for surveillance as the scene is known but which may be too strong an assumption for autonomous driving on board vision.
- MaskRCNN with bottom center of bbox provides a strong baseline for localization. This means the nomography is actually quite accurate. --> For surveillance the camera can be calibrated accurately during installation and then use this MaskRCNN approach.
- The main drawback of the MaskRCNN baseline approach is the low recall. --> I personally feel that this can be improved with more detection data. 
