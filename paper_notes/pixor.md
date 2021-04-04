# [PIXOR: Real-time 3D Object Detection from Point Clouds](https://arxiv.org/abs/1902.06326)

_November 2020_

tl;dr: Voxelize point cloud into 3D occupancy grid for lidar 3D object detection.

#### Overall impression
The paper has a super simple architecture for lidar-only 3D object detection in BEV (3D object localization). This representation is also used in [PIXOR++](pixor++.md) and [FaF](faf.md).

The paper reminds me of the wave of anchor free papers in 2019Q1 such as [CenterNet](centernet.md) and [FCOS](fcos.md). Note that PIXOR still uses decoded box based NMS.

After two years in publication, it is still one of the fastest lidar object detection model out there (33 Hz). It is further improved in [PIXOR++](pixor++.md).

This paper is from Uber's Toronto team, and is superseded by [LaserNet](lasernet.md), also from Uber but from the Pittsburg team. 

#### Key ideas
- Data representation
	- 3D occupancy grid of shape, plus one channel of (accumulatd) reflectance. $L/d_L \times W/d_W \times (H/d_H + 1)$
	- Input 800x700x36 for [0, 70m] x [-40m, 40m] x [-2.5m, 1m].
- Backbone is FPN like, with final resolution 1/4 of original input (4x downsample and 2x upsample). 
- Header
	- Classification: heatmap
	- Regression: 6-ch feature maps. Learning target is $(\cos(\theta), \sin(\theta), \log(dx), \log(dy), \log(w), \log(l))$. Learning targets are normalized by statistics over training set to have zero mean and unit variance. 
- Loss
	- cross entropy (focal loss) for cls
	- Smooth L1 for reg target, and also corner loss for decoded bbox. 

#### Technical details
- BEV vs RV: Drawback of range view (RV, 360 panoramic view as used in [LaserNet](lasernet.md)) include distortion of object size and shape. BEV representation is thus both faithful to physical scale, and avoids overlap of objects to be detected. Also BEV is computationally friendly. 
- Pixels inside GT are positives while outside pixels are negatives. Ignore the pixels inside the donut region [0.3, 1.2] of the original bbox. This helps boost 3.5% AP0.7.
- ATG4D has 5000 training seq, 10Hz and 1200k frames. This roughly translates to 24 seconds per clip. 

#### Notes
- Input data processing can be found [here on github](https://github.com/philip-huang/PIXOR/blob/master/srcs/datagen.py#L244).

