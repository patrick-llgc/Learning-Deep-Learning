# [DA-3Det: Monocular 3D Object Detection via Feature Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540018.pdf)

_August 2020_

tl;dr: Use Domain Adaptation to bridge the gap between pseudo-lidar and real lidar.

#### Overall impression
[DA-3Det](da_3det.md) uses a Siamese network and takes in real lidar and pseudo-lidar data. The difference between the features are penalized. This way [DA-3Det](da_3det.md) learns a general feature based on pseudo-lidar.

Similar ideas to bridge the gap between real and pseudo-lidar has been witnessed in [RefinedMPL](refined_mpl.md), which proposes a way to downsample the dense lidar points to mimic the sparsity of point cloud.

#### Key ideas
- The paper also uses the [Frustum PointNet](frustum_pointnet.md) version of pseudo-lidar due to its simplicity in dealing with point cloud.
- Siamese network with domain adaptation loss (L2 between features).
	- During training process, real-lidar data is also utilized for feature domain adaptation. Only a single image is required during the inference stage.
- Context aware segmentation module: this is simply a pretrained segmentation module that is finetuned online.
	- Pretraining improves performance as compared to unsupervised training with random initialization.
- Domain adaptation is a useful technique that can be applied to mono --> stereo and stereo --> lidar. 

#### Technical details
- Random sampling of lidar point for each object. For object containing smaller numbers of lidar points, sample with replacement (duplication). 

#### Notes
- Questions and notes on how to improve/revise the current work  

