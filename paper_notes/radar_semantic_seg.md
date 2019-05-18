# [Semantic Segmentation on Radar Point Clouds](../assets/papers/schumann2018.pdf)

_May 2019_

tl;dr: Use PointNet++ to perform semantic segmentation of radar point cloud. 

#### Overall impression
The radar point cloud are very sparse, and is usually 2D, lacking the elevation information. However it has one extra important dimension -- Doppler. 

#### Key ideas
- 4D point cloud data (radial distance, azimuth angle, ego-motion compensated Doppler, Radar Cross Section/RCS). 
- Eliminates needs to cluster point cloud and extract features from cluster.
- Grid maps (including occupancy grid map or RCS maps) are good for static scenes but not for moving objects.
- **Feature propagation (FP) module** to propagate sparse label to dense neighborhood.
- Five classes: `ped`, `ped groups`, `cyclists`, `cars`, `trucks`. All others are `static`, including clutter (previously with label `garbage`).
	- `cars` are easily confused with `trucks`.
	- `ped` and `ped groups` are hard to differentiate, as there are noise in human annotation as well.
	- precision for `cars` are not good, only ~68%. Most FP should be `static`.
- **Ego-motion compensated** Doppler has a very large effect on model performance. 

#### Technical details
- For autonomous cars, radar and lidar sensors supplement cameras to maintain functional safety. These additional sensors should not only work complementary but also redundantly.
- In MSG (multiscale grouping module), only spatial info is considered for grouping. Only spatial info (x, y) are used in the grouping.
- Sparse data: 
	- Even at coarse resolution of 1m x 1m, at most 6% of the grid will have non-zero values.
	- Only 2% to 3% of all points are non-static objects. 
- Each point in moving object is dropped out with random prob from [0, 0.3].
- 500 ms worth of data is accumulated. But only 3072 data points are used (if more, then static points are dropped; if less, then points are resampled). During inference, every 3072 points were passed though network in the chronological order so no over- or under-sampling.
- Moving vs Doppler: Doppler is not absolute indicator of moving objects. Many static objects also have non-zero Doppler due to error in odometry, sensor misalignment, time sync error, mirror effects or other sensor artifacts. On the other hand, bottom of a rotating car wheel or pedestrian walking radially also does not have doppler effect. 

#### Notes
- **Feature propagation module** should be very useful in propagating sparse labels to dense data. Need to read [PointNet++](pointnet++.md) again.
- Plotting Range-Cross range map with Doppler as color legend helps quite a lot in **human annotation**.
- Doppler signal needs to be **motion compensated**.
