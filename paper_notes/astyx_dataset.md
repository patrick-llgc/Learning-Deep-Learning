# [Astyx dataset: Automotive Radar Dataset for Deep Learning Based 3D Object Detection](https://www.astyx.com/fileadmin/redakteur/dokumente/Automotive_Radar_Dataset_for_Deep_learning_Based_3D_Object_Detection.PDF)

_January 2020_

tl;dr: Dataset with radar data from proprietary high resolution radar design.

#### Overall impression
Active learning scheme based on uncertainty sampling using estimated scores as approximation.

#### Key ideas
- Radar+camera sees more clearly than lidar+camera, for far away objects and for pedestrians. --> However even with radar, the recall is only ~0.5. Too low for real-world application.

#### Technical details
- Cross sensor calibration has two steps: camera lidar 2D-3D with checkerboard, and radar lidar 3D-3D relative pose estimation.
- Annotation has "invisible" objects as well associated via temporal reference, but invisible in camera and lidar. 

#### Notes
- [Dataset](https://www.astyx.com/development/astyx-hires2019-dataset.html)
- [Estimation of height](https://sci-hub.tw/10.1109/RADAR.2019.8835831) in this dataset