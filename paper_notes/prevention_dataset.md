# [The PREVENTION dataset: a novel benchmark for PREdiction of VEhicles iNTentIONs](10.1109/ITSC.2019.8917433)

_January 2020_

tl;dr: Dataset for cut-in and other lane-change prediction.

#### Overall impression
Describe the overall impression of the paper. 

#### Key ideas
- High accuracy lateral movement of the vehicles are critical for the lane change prediction task. Symmetrically placed keypoints are traced (like BPE) automatically, with Median Flow. The tracking process is supervised.
- Relative position of ego-vehicle wrt the road surface is useful to correct BEV pitch or height. The ground plane coefficients are computed with RANSAC with lidar point cloud.
- Event types: Cut-in/Cut-out/Lane-change/zebra-crossing

#### Technical details
- Cameras are externally triggered by lidar at 10 Hz.
- On average, one lane change event in every 20 second. 
- Lane detection and tracking in BEV image with moving objects removed.

#### Notes
- We need some metric to measure the prediction performance!

