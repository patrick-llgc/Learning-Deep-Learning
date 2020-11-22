# [LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving](https://arxiv.org/abs/1903.08701)

_November 2019_

tl;dr: A real-time probabilistic lidar detector based on RV (range view) that Models aleatoric uncertainty.

#### Overall impression
The introduction of probabilistic object detection boosts the performance by nearly 4%. This work is succeeded by [LaserNet KL](lasernet_kl.md) by modeling the noise/uncertainty in label as well.

The paper intentionally only predicts aleatoric uncertainty as there is no efficient way to compute epistemic uncertainty.

The RV view of point cloud is dense and is the native representation of lidar. Projecting to 3D space or BEV leads to sparse representation and more computation.

I feel the authors started with a RV-based lidar detector, but only added probabilistic object detection as a novelty to boost performance.

#### Key ideas
- Predict a bbox per point. This is similar to the idea of [Point RCNN](point_rcnn.md). 
- Model a **multimodal** distribution with a mixture model (Laplacian) for cars (3 modes). This essentially has 3 mean and 3 std. During training only the most likely distribution gets updated. This is similar to training anchors. 
- Detection on RV leads to computational efficiency (natively dense) and it can reach 30 ms, or only 12 ms forward only. (Not that fast as compared to the contemporary [PointPillars](point_pillars.md) reaches 115 Hz).
- Probabilistic object detectors are harder to train. On KITTI the calibration curve looks bad, and only on ATG4D (Uber's proprietary dataset, about 100 times larger) the detector learns a good calibration curve and beats other SOTA methods. 


#### Technical details
- Range view: lidar produces a cylindrical range image. This is very similar to the spherical view. --> See [pillar OD](pillar_od.md) for cylindrical view which is an improvement over this.
	- rows: laser id (elevation)
	- cols: azimuth
	- depth: r
- Mean shift clustering
	- mean and std are averaged with others that are grouped into the same cluster by mean shift clustering.
- Adaptive NMS: a bit like soft NMS. Instead of hard suppressing, increase the uncertainty std (similar to soft NMS which suppresses the confidence score)

#### Notes
- Mean shift clustering
	- [Mean shift clustering](https://spin.atomicobject.com/2015/05/26/mean-shift-clustering/) is based on the idea of KDE, and tries to place each point to the nearest peak (mode). It is one of the mode-seeking algorithm. However it is computationally intensive.
	- In comparison, k-means can be done more efficiently, but it cannot handle non-linear data well.
	- There is a way to combine the two together, first by doing k-means and then do mean shift. See [this blog post](http://jamesxli.blogspot.com/2012/03/on-mean-shift-and-k-means-clustering.html). It is efficient and can handle non-linear data well. 
	- In a sense, the paper first quantize into bins of 0.5x0.5, then perform mean shift clustering, very similar to the above method.