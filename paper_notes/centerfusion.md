# [CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection](https://arxiv.org/abs/2011.04841)

_November 2020_

tl;dr: Early fusion of camera and radar based on [CenterTrack](centertrack.md).

#### Overall impression
This is one of the best paper on the topic of camera and radar fusion up till 2020Q4. 

Data association between radar and camera is critical for radar and camera association. The naive solution of inspecting if a radar pin falls into one bbox is not robust as there is not a one-to-one mapping between radar detection and bbox. The paper proposed a frustum-based association method, consisting of frustum building and pillar expansion.

#### Key ideas
- **Pillar expansion**: Radar lack elevation information and therefore each radar point is extended to a fixed sized pillar. 
- **Frustum association**: 
	- CenterFusion build upon mono3D with [CenterNet](centernet.md), then creates a ROI frustum.
	- If the 3D **frustum** created by the bbox has overlap with the 3D **pillar** created by the radar pin, then they are associated.
- Splat radar features onto images: After association, every radar pin generate 3 channel heat map, at location of the bbox. (depth, x and y of radial velocity).
	- For overlapping region, pick the closer object.
	- The 3-ch heatmaps are then concatenated to the feature maps, then use secondary head to predict depth and orientation and velocity.

#### Technical details
- Cons of Lidar vs radar
	- Lidar and camera are both sensitive to adverse weather conditions
	- No speed information --> FMCW lidars (Strobe, Blackmore, Aeva) will most likely fix this.
- Cons of Radar 
	- lack of dataset containing radar data
	- significantly more sparse than lidar, aggregation over timeframe would lead to system delay
	- vertical measurement is almost non-existent
- Early fusion vs late fusion
	- Early fusion is sensitive to spatial or temporal misalignment of the data
	- Late fusion is more flexible and extendable to new sensing modalities, but does not exploit the full potential
- Depth regression: log(depth) + L1.

#### Notes
- Code on [github](https://github.com/mrnabati/CenterFusion).

