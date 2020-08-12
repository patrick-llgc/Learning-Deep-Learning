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
- There are several other works which also uses road markings for mapping and localization
	- [LaneLoc: Lane Marking based Localization using Highly Accurate Maps](http://www.cvlibs.net/projects/autonomous_vision_survey/literature/Schreiber2013IV.pdf) <kbd>IV 2013</kbd>: uses curbs and road markings.	- [Light-weight Localization for Vehicles using Road Markings](http://www.ananth.in/PubsByYear_files/Ranganathan13iros.pdf) <kbd>IROS 2013</kbd>: use multiple corners of the road arrows for localization. Corner detection and segmentation are performed by snake and FAST. --> But for pose we can use dead reckoning with IMU in short range (up tp 1km).
	- [Submap-Based SLAM for Road Markings](https://www.mrt.kit.edu/z/publ/download/2015/rehder_iv15.pdf) <kbd>IV 2015</kbd> [Honda]: uses odometry and camera images for loop closure 
	- [Monocular Localization in Urban Environments using Road Markings](http://bheisele.com/Lu_HDL_IV2017.pdf) <kbd>IV 2017</kbd> [Honda]: uses epipolar geometry and odometry

