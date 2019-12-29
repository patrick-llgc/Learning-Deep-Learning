# [Vehicle Detection With Automotive Radar Using Deep Learning on Range-Azimuth-Doppler Tensors](http://openaccess.thecvf.com/content_ICCVW_2019/papers/CVRSUAD/Major_Vehicle_Detection_With_Automotive_Radar_Using_Deep_Learning_on_Range-Azimuth-Doppler_ICCVW_2019_paper.pdf)

_November 2019_

tl;dr: Deep radar perception on radar FFT data.

#### Overall impression
Directly regress oriented bbox (maybe it is even upright bbox) on front radar. This is similar to [deep radar detector](deep_radar_detector.md). 
This work is succeed by Qualcomm's NIPS 2019 paper on [camera and radar fusion](radar_camera_qcom.md).

Limitation: The dataset only contains highway driving, and only have upright bbox.

The paper reiterated that **when a transformation is mathematically known, it is easier to code it into the neural network**.


#### Key ideas
- Raw ADC signal is converted to range-azimuth-doppler tensor by 3D FFT. The phase information is discarded and only amplitude is feed into the neural network.
- RA model: doppler channel power is summed up. Similar to RetinaNet/SSD. 
- RAD model: three encoders, then generate meshgrid to duplicate along one channel to match the size of three feature maps. The feature maps are then concatenated along channel, decoded and converted to Cartesian before detection head. --> polar2cartesian layer is better placed earlier in the network (but not necessarily directly on the input) as pointed out in [camera and radar fusion](radar_camera_qcom.md).
- Using doppler info does not conclusively help predict velocity better, but with temporal information does.
- Radar based system is less sensitive to distance of the target (but only up to 45 meters).

#### Technical details
- The doppler cues does not help too much (~1% increase)
- Data acq setup:
	- Max range: 46.8 m
	- Range resolution: 0.094 m --> BW = 1.6 GHz
	- Max Doppler: +/- 2.5 m/s
	- Doppler resolution: 0.25 m/s
	- Azimuth resolution: 5 deg --> 2/N rad, N = 32
	- Frame rate: 125 Hz
- only the output from a single layer is used as there is no scale variance. 
- Regresses the magnitude and direction of velocity.
- A sequence length of 10 frames are used for LSTM model, but showed little improvement (1% mAP).
- only data augmentation is horizontal flipping. 

#### Notes
- Questions and notes on how to improve/revise the current work  

