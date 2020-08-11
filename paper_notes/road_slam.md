# [Road-SLAM : Road Marking based SLAM with Lane-level Accuracy](https://www.naverlabs.com/img/autonomousDriving/intelligence/dissertation/Road-SLAM_Road%20Marking%20based%20SLAM%20with%20Lane-level%20Accuracy.pdf)

_July 2020_

tl;dr: Use RSM for point cloud map building and localization.

#### Overall impression
Road markings is susceptible to visual aliasinig for global visualization. Only six informative classes (dashed lanes, arrows, road markings, numbers, crosswalk) of road markings are considered to avoid ambiguity.

Road-SLAM can achieve cm accuracy.

#### Key ideas
- Recognize places with only road markings, less sensitive to environmental changes (lighting, time, surroundings, etc). 
- IMU is the backbone, and gives accurate prediction within km level. 
- Sub-map is created when a road marking is detected, and stored and used for loop closure. 
- g-ICP based 

#### Technical details
- GPS signal suffers from multipath and blackout issue, especially in complex urban canyon. And it is only meter level accuracy, not good enough for lane level localization.
- Classification through ESF (ensemble of shape function)

#### Notes
- [Vehicle Localization with Lane Marking papers](https://sites.google.com/site/yorkyuhuang/home/tutorial/autonomous-driving-1/vehicle-localization-with-lane-markings) from 黄浴

