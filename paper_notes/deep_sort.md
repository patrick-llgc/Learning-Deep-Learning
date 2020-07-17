# [deep-sort: Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

_September 2019_

tl;dr: use a combination of appearance metric and bbox for tracking.

#### Overall impression
The problem with [sort](sort.md) is the frequent ID switches as sort uses a simple motion model and does not handle occluded tracks well.

Deep sort uses the appearance features to track objects through longer periods of occlusion. In reality the cost only consists of appearance metrics, although bbox distance is used as a gating process.

[deepSORT](deep_sort.md) extracts features from image patches, while [PointTrack](pointtrack.md) extracts features from 2D point cloud. 

#### Key ideas
- Same as sort:
	- Kalman filter with constant velocity motion and linear observation model where bounding box as direct observations of the object state. --> maybe take a look at the Bayesian object detection?
- For each track we track the age of the track where it is not associated
- **Difference from sort**:
	- instead of using IOU metric, a weighted sum of Mahalanobis distance (between predicted bbox and measurement) with appearance distance, between i-th track and j-th detection
	- the bbox distance $d^{(1)} = (d_j - y_i)^T S_i^{-1} (d_j - y_i)$, where (y, S) is the i-th track in measurement space, and d is the j-th detection.
	- the appearance distance $d^{(2)} = min\{1 - r_j^T r_k| r_k \in R_i \}$
	- Cost matrix $c_{i,j} = \lambda d^{(1)} + (1-\lambda) d^{(2)}$ (in practice \lambda is set to 0 where there is substantial camera motion)
	- Gate matrix $b_{i,j} = \mathbb{1}[d^{(1)} \le t^{(1)}] \times \mathbb{1}[d^{(2)} \le t^{(2)}]$ (even though \lambda=0, the Mahalabobis distance is still used to discard impossible measurements/detections)
- Prioritize more recently seen objects (smaller age)
- Deep appearance feature: l2 normalized feature, trained with triplet loss on re-ID database. See paper from the same author [at WACV 2018](https://arxiv.org/pdf/1812.00442.pdf) (triplet loss is still SOTA on Re-ID).

#### Technical details
- ID switch is among the smallest and around half of sort.
- However the FP is much higher than sort, mainly due to maximum age A_max = 30 frames. 

#### Notes
Code available at this [repo](https://github.com/nwojke/deep_sort).

