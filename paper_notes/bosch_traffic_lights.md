# [A Deep Learning Approach to Traffic Lights: Detection, Tracking, and Classification](https://doi.org/10.1109/ICRA.2017.7989163)

_July 2019_

tl;dr: Traffic light dataset with still images as training and video as tracking. No traffic light association is annotated.

#### Overall impression
Train with still images, and test is on video with tracking.

This is quite different with turn signal light detection. 

- Turn signal lights (TSL) flicker at much higher rate (1-2 Hz) and have strong temporal features. Traffic lights are most static, with occasional change.
- TSL is only on for brief period of time, but traffic lights are on all the time. 
- TSL has a lot more visual variety than traffic lights. 
- However, traffic lights is more safety critical and requires higher recall/precision than TSL. It is also hard to associate traffic lights with the ego lane and ego vehicle.


#### Key ideas
- A tracker is used to boost the performance of detection, due to occlusion or off-state due to camera sampling rate. It has a motion model that uses IMU (not available in dataset). It is essentially doing prediction and tracking at the same time.
- Each bounding box is trained with predicted confidence supervised by IoU. 
- The network always predict a confidence score supervised by 

#### Technical details
- Major FP: lamps, decorations, reflections, taillights.
- 1280x720 images, training annotated at 2 fps, test annotated at 15 fps.
- Many traffic lights appear to be off due to difference in camera sampling freq and traffic light refresh rate.
- Only 3 448x448 patches are used for inference. 
- Crop context out for classification --> this will be taken care of directly from the feature map?
- Four categories: "green", "yellow", "red" and "off"

#### Notes
- Use RetinaNet to improve the single stage object detector?

