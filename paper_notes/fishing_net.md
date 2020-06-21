# [FISHING Net: Future Inference of Semantic Heatmaps In Grids](https://arxiv.org/abs/2006.09917)

_June 2020_

tl;dr: Detached model to perform domain adaptation sim2real.

#### Overall impression
Two stage model to bridge domain gap. This is very similar to [GenLaneNet](gen_lanenet.md) for 3D LLD prediction. 

The first stage model already extracted away domain-dependent features and thus the second stage model can be used as is.

![](https://cdn-images-1.medium.com/max/1280/1*mmAdzMVKxAjP0CvSO618dw.png)

#### Key ideas
- The first stage generates the pseudo-lidar point cloud in BEV. 
- The second stage converts pseudo-lidar point cloud to BEV segmentation.
- During inference, only finetune first stage. Use second stage as is. 
- The simulated segmentation GT can be obtained in abundance with simulator such as CARLA. 

#### Technical details
- Summary of technical details

#### Notes
- [talk at CVPR 2020](https://youtu.be/WRH7N_GxgjE?t=1554)

