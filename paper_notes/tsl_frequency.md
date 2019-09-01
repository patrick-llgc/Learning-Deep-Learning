# [Will this car change the lane? - Turn signal recognition in the frequency domain](https://ieeexplore.ieee.org/document/6856477/)

_August 2019_

tl;dr: Detect TSL change in frequency domain (FFT).

#### Overall impression
The paper did not use DL techniques, but rather based on CV and conventional ML method. The ML method may also be replaced by DSP method. 

The authors used this method partially due to the difficulty in data collection.

#### Key ideas
- TSL flashing frequency 1.5 +/-0.5 Hz according to EU/US regulation. The paper detects no later than 3 flashes (2 seconds detection window). 
- State machine: Time smoothing takes one sequence of 2/3 second. If 50% of the evaluated frame is indicated flashing then the tracklet is indicated as flashing.
- Pipeline: 
	- MOD detection
	- detect light spot
	- extract descriptor
	- transform descriptor (FFT)
	- Classify each frame of the tracklet behavior
	- smooth over time
- Feature: find pixel clusters first, then for each frame of the past 2 seconds, the average intensity of all those pixels is used as the signal. With 16 fps, 30 frames are used for predicting the labels.
- Feature transformation: with FFT 

#### Technical details
- Background extraction with disparity map (stereo camera system required for data acquisition)
- **5%**: ~50 out of 2500 vehicles has blinkers on, and only 25 or so are trackable in object detection. The rest are visible but missed by object detection and tracking algorithm.
- **3/min**: One of the challenge of TSL recognition is the scarcity of labels. On average only three cars has TSL blinking in dashcam videos per minute. 
- Evaluation: 
	- tracklet level false positive vs recall
	- performance breakdown with distance
	- recognition rate with flashing times
	- Average time delay (~1.55 s) and time delay histogram

#### Notes
- This method is very practical. However I feel the ML part is forced novelty and can also be replaced with some DSP method. 

