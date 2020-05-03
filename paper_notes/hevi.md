# [Egocentric Vision-based Future Vehicle Localization for Intelligent Driving Assistance Systems](https://arxiv.org/abs/1809.07408)

_May 2020_

tl;dr: Egocentric/first person vehicle prediction.

#### Overall impression
The paper introduced HEVI (Honda egocentric view intersection) dataset. 

First-person video or egocentric data are easier to collect, and also captures rich information about the objects performance. 

However the front camera has a narrow FOV and tracklets are usually short. The paper selects tracklets that are 2 seconds long. Use 1 sec history and predict 1 second future. 

The inclusion of dense optical flow improves results hugely. Incorporation of future ego motion is also important in reducing prediction error. Note that the future ego motion is fed as GT. During inference the system assumes future motion are from motion planning.

#### Key ideas
- Summaries of the key ideas

#### Technical details
- Motion planning is represented in BEV, with 2 DoF translation and one DoF of rotation (yaw).
- HEVI classifies tracklets as easy and hard. Easy can be predicted with a **constant acceleration** model with lower than average error. 

#### Notes
- Questions and notes on how to improve/revise the current work  

