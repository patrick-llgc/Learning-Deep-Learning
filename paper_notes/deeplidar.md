# [DeepLiDAR: Deep Surface Normal Guided Depth Prediction for Outdoor Scene from Sparse LiDAR Data and Single Color Image](https://arxiv.org/pdf/1812.00488v2.pdf)

_June 2019_

tl;dr: Depth completion from a single RGB image and sparse depth (from lidar and low-cost lidars or one-line lidars). 

#### Overall impression
The idea should benefit similar projects for radar+camera sensor fusion. The main idea is **to take a sparse but accurate depth from a low-cost lidar and make it dense with the help of an aligned color image**. However the paper only explored uniform sparsity, not structured sparsity (missing lines of depth completely) which is perhaps more relevant for cheap lidars. 

The paper heavily builds on the [indoor depth completion paper](https://arxiv.org/pdf/1803.09326.pdf). It is in a sense color guided depth impainting.

#### Key ideas
- Two pathways: color pathway directly regresses the depth, and the normal pathway regresses surface normal first and then convert to depth map. 
	- In the color pathway
		- input: rgb, sparse depth, binary mask
		- output: **confidence mask**, dense depth 1, attention map 1
	- In the normal pathway
		- Stage1
			- input: rgb, sparse depth, binary mask
			- output: surface normals
		- Stage2
			- input: surface normal, **confidence mask**, sparse depth
			- output: dense depth 2, attention map 2
- Even for lidar, the warped depth projected onto camera have mixed depth information due to camera/lidar offset and optical occlusion. Therefore a confidence mask is used to indicates the reliability of the input sparse depth. The confidence map is regressed from the color pathway and used to guide the depth prediction in the normal pathway.
- Attention mechanism is used to integrate the predictions from two pathways into a single prediction. In addition to the depth map, an attention map (weight map) is also predicted and fed into a softmax.

#### Technical details
- Early fusion of feeding sparse depth and camera images directly into CNNs will lead to artifacts (the authors did not specify source). Refer to the [MV3D](mv3d.md) paper for review of early fusion, late fusion and deep fusion.
- Deep completion unit (DCU) is late fusion, very similar to that proposed in [depth completion and semantic segmentation](https://arxiv.org/abs/1808.00769) paper. The encoder extract features separately and only combines them in the decoder. 
- Outdoor scenes are more susceptible to noise in surface normals, increasingly so in distant regions. 
- The paper uses surface normals as intermediate representations (as it is easier to estimate per [the indoor paper](https://arxiv.org/pdf/1803.09326.pdf)).
- Ablation test:
	- Even without the normal pathway, the results are not bad at all. 
	- Early fusion also works, but not as good as late fusion.
	- With more sparcity (as few as 72 points per depth image), the RMSE increased from 0.7m to 2m RMSE, still better than conventional methods with full sparse data. 

#### Notes
- There is a KITTI depth completion benchmark. How did they get the GT?
- Why is surface normal easier to estimate than absolute depth?
- Maybe convert RGB to point cloud and adjust depth in point cloud space yields better results?
- How do unsupervised depth estimation evaluate the performance?
- How was the surface normal GT obtained in KITTI? From GT dense depth map by local plane fitting.
