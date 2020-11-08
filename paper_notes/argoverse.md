# [Argoverse: 3D Tracking and Forecasting with Rich Maps](https://arxiv.org/abs/1911.02620)

_November 2020_

tl;dr: Map data helps with 3D tracking and motion forecasting.

#### Overall impression
This dataset is one of a kind and focuses on tracking and motion forecasting (of other traffic agents other than ego car). The only other dataset that focuses on this is Lyft 3D dataset. 

The original of the ego is center of the real axel. All lidar data is provided in the ego vehicle coordinate system.

Given the multimodal nature of the motion forecasting problem,. It might not be fair to evaluate against a single GT.

The paper used a CV-based lidar detection method. This is more traditional method used since the DARPA challenge. 

#### Key ideas
- Argoverse provides 2 Datasets
	- 3D tracking: 113 segments varying from 15 to 30 seconds. 
	- Motion forecasting: 300,000 5-second tracked scenario, collected at 10 Hz. Predict 30 timestamps into future with 20 given. 
- Map
	- Vector map of lane centerlines and their attributes (if a lane is in intersection or traffic control measures, lane turn direction, and uuids, predecessor and successors) --> How is this organized? For merges and splits?
	- rasterized map of ground heights
	- rasterized map of drivable area

- HD map helps with 3D tracking
	- ground points removal
	- **orientation snapping** to lanes. Vehicle orientation rarely violate lane directions.
- HD map helps with motion forecasting
	- Use drivable space
	- Lane graph
- **Curvilinear centerline coordinate system** wrt the centerline makes the trajectory forecasting much easier. 
- Lidar-based object detection with CV methods. 
	- DBSCAN for clustering
	- MaskRCNN for verification and discard any non-vehicle clusters.
- The dataset can also be used for *map automation*.

#### Technical details
- Two simplest motion forecasting baselines: Nearest neighbors, and constant velocity model.
- Mined 1000 hour of fleet logs for **interesting** real-world trajectories. --> this can be done with onboard miner.
	- Every 5 seconds, assign one "interesting" score to the sequence. 2.5 seconds overlap for two consecutive sequence.
	- Left/right turn, left/right lane change, at intersection, having high variance in velocity and visible for long duration.
	- At least 2 interesting tracks to save the sequence for forecasting jobs.
- Lane centerline instead of road centerline. 
- In 3D tracking, absolute distance between centroids are used for association. This may be one of the reason where 3D IoU is not used in [Nuscene](nuscenes.md) AP calcuation.
- Vehicle classes:
	- Emergency vehicle: when siren on, gains right of way in all situations.
	- Large vehicle: more than 4 wheels and cannot fit into normal garage.
- Desensitization: faces and license plates are procedurally blurred.

#### Notes
- CV based lidar detection 
	- [LIDAR-based 3D Object Perception](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.677.2534&rep=rep1&type=pdf), 2008
	- [Towards Fully Autonomous Driving: Systems and Algorithms](https://www.ri.cmu.edu/wp-content/uploads/2017/12/levinson-iv2011.pdf), 2011. This also includes a traditional TFL detection algorithm.

