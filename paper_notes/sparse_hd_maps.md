# [Exploiting Sparse Semantic HD Maps for Self-Driving Vehicle Localization](https://arxiv.org/abs/1908.03274)

_August 2020_

tl;dr: Use lane graph and traffic sign to build sparse HD map on highways to perform cm-level localization.

#### Overall impression
The sparse HD map uses lane graphs (LLD) and traffic sign (TSR) and only takes 0.03%, orders of magnitude smaller storage space than traditional HD maps.

The proposed method aims to tackle localization on highways where loop closure cannot be used. 

Sparse HD map does not require detailed knowledge about the appearance of the world (dense geometry and texture) --> this leads to better generalization.

It is interesting to see that 

- $LLD + TSR + GPS  \gtrsim LLD + GPS \gt LLD + TSR$
- with GPS, the precision is improved quite a bit. 
- Without TSR, the performance drops a little bit. 
- LLD + TSR can barely achieve cm mean lateral accuracy, but 99% lateral error is too large (1.4 m) to be used for autonomous driving.

#### Key ideas
- Traffic sign provide info in longitudinal direction, while lanes help avoid lateral drift
- The proposed method only cares about 3DoF localization (x, y, yaw) wrt the map, instead of a full 6 DoF.
- Perception with DL
	- LLD: predict truncated inverse distance transformation.
	- TSR: 5k images with bbox annotation
- Mapping: TSR to get bbox, then use frustum to get 3D location of TSR.
- The pose estimation is formulated as a histogram filter
	- Bel: Belief, a prob distribution over the state of x. 
	- posterior proba Bel(x)
	- $Bel(x)=P_{lane}(Sensor Signal | x) P_{sign}(Sensor Signal| x) P_{GPS} Bel_{t|t-1}(x|\chi)$
	- Discretize the continuous search into a histogram filter.
- Evaluation: 
	- Lateral
	- Longitudinal
	- Smoothness
- Baseline:
	- Dead-reckoning (dynamics): worst
	- GPS alone
	- INS (GPS + IMU) --> ? not explicitly explained
	- LLD/TSR + GPS + IMU

#### Technical details
- Resolution: 5 cm / pixel.
- IMU drift: 0.4%. TSR may happen once per km, thus 4 m drift. 
- Localization GT is obtained with a a high precision ICP based offline Graph SLAM using high def pre-scanned scene geometry.
- Geometric approaches suffer in the presence of repetitive patterns.
- Loop closure can be used to fix the accumulated errors. Loop closure is less useful on highways where trajectories are unlikely to be closed. This makes drift an even harder problem to overcome
- Traffic signs are semantic landmarks that are sparsely yet systematically present in cities, rural areas and highways. 
- This work ignores the effects of suspension, unbalanced tires and vibration.

#### Notes
- [talk sides](https://siegedog.com/assets/pdf/talks/BARSAN-IoanAndrei-2019-IROS-SparseHDMaps-Talk-v6.pdf)
- [A Low Cost Vehicle Localization System based on HD Map](https://medium.com/@yuhuang_63908/a-low-cost-vehicle-localization-system-based-on-hd-map-50d464ba5d8f)
- INS: inertia nav. sys.
- geo-registered: geo-tagged, marked with lat/long info
- From the ablation study, it looks like without GPS, the system cannot achieve cm accuracy. --> This sounds like a deal-breaker for urban driving, but this may help quite a bit when we use markings on the road, instead of objects in the air (TSR). Also the signs will be denser in urban region.