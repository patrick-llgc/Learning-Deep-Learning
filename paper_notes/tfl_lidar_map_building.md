# [Traffic Light Recognition Using Deep Learning and Prior Maps for Autonomous Cars](https://arxiv.org/abs/1906.11886)

_November 2020_

tl;dr: Build traffic light map offline with lidar and use it to guide online perception.

#### Overall impression
This paper is not fancy but is quite practical with many engineering details. 

The main challenges for TFL

- adverse condition
- early recognition
- recognition in diff illumination settings (night)

Maps can be used to only trigger TFL recognition module near intersections. This saves computation and avoid false positives but this unfortunately disables map building capability from fleet cars. 

#### Key ideas
- Making use of prior maps is necessary because until the present moment there is no clear algorithm or machine learning method that can robustly identify which traffic lights are relevant using only image data. --> See [Deep Lane Association](deep_lane_association.md).
- Traffic lights that share the same control semantics can be grouped together thus their redundancy can come to use when the state of one of the them cannot be determined.
- The car should stop at the stopline when the color is red/yellow/off.
- TFL map building by projecting lidar points to camera, and aggregate points inside the TFL bbox. The centroid of the cluster is used as the 3D position of the TFL in the map.

#### Technical details
- RoI: 3D position from map as the center for a sphere with 1.5m radius. Any BB that has its center outside of all projected sphere are discarded right away.
![](https://images.deepai.org/converted-papers/1906.11886/x2.png)

- Reducing detection threshold resulted in more false proposals from the detector along the frames, but this is not an issue as most of the FPs are filtered out by maps.

#### Notes
- Questions and notes on how to improve/revise the current work  

