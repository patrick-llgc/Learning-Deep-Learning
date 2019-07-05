# [Pseudo-LiDAR++: Accurate Depth for 3D Object Detection in Autonomous Driving](https://arxiv.org/pdf/1906.06310.pdf)

_June 2019_

tl;dr: Improve depth estimation of pseudo-lidar with stereo depth network (SDN) and sparse depth measurements on "landmark" pixels with few-line lidars.

#### Overall impression
This is exactly what I wanted to do when I read [pseudo-lidar](pseudo_lidar.md). However we could still explore the idea with radar data. 

Pseudo-lidar bridged half of the gap between RGB-based and lidar based 3D object detection but does not perform well for far-away object. Pseudo-lidar++ uses sparse 3D measurement to de-bias the depth estimation.

The Uber ATG group also publishes several papers ([ContFuse](contfuse.md), [MMF](mmf.md)) on this idea, although not as explicit as the pseudo-lidar paper or this one.

#### Key ideas
- The depth estimation error from monocular images are not random but rather systematic. 
- **Idea #1**: SDN (stereo depth network) to address the issue that neural network designed to optimize disparity over-emphasize nearby objects due to reciprocal transformation. $\delta Z \propto Z^2 \delta D$
	- The idea of optimizing distance loss instead of disparity is rather staightforward.
	- The change of depth cost volume instead of disparity cost volumn (as per [PSMNet](https://arxiv.org/pdf/1803.08669.pdf), SOTA for stereo depth estimation).
- **Idea #2**: Depth correction algorithms (GDC: graph depth correction)
	- two step optimization process: learn the depth relationship of different neighbors first $W$, then use the relationship to constrain the depth changes. 

#### Technical details
- The depth estimation from stereo or other image is imprecise in nature that the horizontal coordinates have to be quantized. 
- At IOU 0.5, with the aid of only 4 lidar beams, PL++ is boosted to a level comparable to models with 64-beam lidar.
- 4-beam lidar performs well on faraway objects, while replacing the points (not the neighbors) hinders detection of faraway objects. 
- Synthesized few line lidar data based on angular separation spec of Ibeo lidar, instead of just selecting 2 or 4 lines randomly.

#### Notes
- Apply this to radar data (single line lidar?) and see if this improves results.
- Need to read PSMNet. This seems to regress disparity via classification?
	- This seems to be originating from [PMSNet](https://arxiv.org/pdf/1803.08669.pdf) (CVPR 2018) and [GC-Net: End-to-End Learning of Geometry and Context for Deep Stereo Regression](https://arxiv.org/abs/1703.04309) (ICCV 2017).

