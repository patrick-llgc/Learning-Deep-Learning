# [Traffic Light Mapping, Localization, and State Detection for Autonomous Vehicles](http://driving.stanford.edu/papers/ICRA2011.pdf)

_December 2020_

tl;dr: Mapping and online detection of TFL.

#### Overall impression
Traffic light state perception is the key. 

Even in non-autonomous vehicles, traffic light state detection would be beneficial alerting inattentive drivers to changing light status and making intersections safer. --> TLA

The key to safe operation is the ability to handle common failure cases, such as FP or transient occlusion for camera based TFL perception.

#### Key ideas
- TFL tracking and triangulation according to [Optimal Ray Intersection For Computing 3D Points From N-View Correspondences](http://gregslabaugh.net/publications/opray.pdf).
- Temporal filtering with **hysterisis**: only change light state when several frames of identical color detection have occurred sequentially. This adds a fraction of a second latency but the response time still matches or is better than that of a human. 

#### Technical details
- Camera with fixed camera 
- Coordinate transformation: world, vehicle, camera and object
![](https://cdn-images-1.medium.com/max/1600/1*9ACrh-29WEJrCFxgJA_YlQ.png)
- Traffic light color hue histogram
![](https://cdn-images-1.medium.com/max/1600/1*TKJ1EdZY-pzYsnC3M0J1yw.png)

#### Notes
- Questions and notes on how to improve/revise the current work  

