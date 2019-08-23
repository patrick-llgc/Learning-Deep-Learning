# [Distance Estimation of Monocular Based on Vehicle Pose Information](https://iopscience.iop.org/article/10.1088/1742-6596/1168/3/032040/pdf)

_August 2019_

tl;dr: Use IMU for online calibration to get real time pose. Use radar for GT

#### Overall impression
The paper is a short technical report. 

#### Key ideas
- The main idea is to estimate distance based on the bottom line of the vehicle bbox. 
- For this, accurate and online calibrated **roll and pitch** are needed (yaw will not chance the row position of the car in the image). 
- radar is used to acquire GT.
- The method performs really well for **vehicles up to 30 m** (with errors of up to 0.5 m).

#### Technical details
- Camera, radar and IMU run at 20 Hz. 



