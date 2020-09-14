# [BEV-Seg: Bird’s Eye View Semantic Segmentation Using Geometry and Semantic Point Cloud](https://arxiv.org/abs/2006.11436)

_June 2020_

tl;dr: Detached model to perform domain adaptation sim2real.

#### Overall impression
Two stage model to bridge domain gap. This is very similar to [GenLaneNet](gen_lanenet.md) for 3D LLD prediction. The idea of using semantic segmentation to bridge the sim2real gap is explored in many BEV semantic segmentation tasks such as [BEV-Seg](bev_seg.md), [CAM2BEV](cam2bev.md), [VPN](vpn.md).

The first stage model already extracted away domain-dependent features and thus the second stage model can be used as is.

The GT of BEV segmentation is difficult to collect in most domains. The simulated segmentation GT can be obtained in abundance with simulator such as CARLA. 

![](https://cdn-images-1.medium.com/max/1280/1*mmAdzMVKxAjP0CvSO618dw.png)

#### Key ideas
- **View transformation**: pixel-wise depth prediction
- The first stage generates the pseudo-lidar point cloud, and render it in BEV.
	- This is incomplete and may have many void pixels.
	- Always choosing the point of lower height.
- The second stage converts the BEV view of pseudo-lidar point cloud to BEV segmentation.
	- Fills in the void pixels
	- Smooth already predicted segmentation
- During inference, only finetune first stage. Use second stage as is. 


#### Technical details
- Summary of technical details

#### Notes
- [talk at CVPR 2020](https://youtu.be/WRH7N_GxgjE?t=1554)

