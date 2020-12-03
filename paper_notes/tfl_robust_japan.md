# [Robust Traffic Light and Arrow Detection Using Digital Map with Spatial Prior Information for Automated Driving](https://www.mdpi.com/1424-8220/20/4/1181)

_September 2020_

tl;dr: Combines DL and conventional method and HD map prior info for robust traffic light detection.

#### Overall impression
This paper looks highly practical and engineering focused. It provided many benchmarks to compare against. It is limited to traffic light in Japan, which is perhaps a well regulated market without too many out-of-spec traffic lights. Still, there exist many challenges caused by the nature of the problem, such as distance, lighting condition, etc.

The paper demonstrated flat F1 values up to 150 m (5 pixels).

#### Key ideas
- Combine Yolo, SURF keypoint and blob object detection.
	- SURF: detecting blobs by det(H)
- **HD Map** should contain info about traffic light
	- 2D TL position (lat, long, heading orientation), 3DoF. --> Only traffic light within a certain range and orientation to ego vehicle are projected back to image.
	- format (horizontal, vertical, 1x3, 3x2, etc)
	- No height information, only approximate height info during online inference according to country (~5.0 meter in Japan) or set a wide recognition area in image when height is unknown.
- Highlighted image generation in HSV space
	- Normalize the brightness Value to emphasize the lighting
	- Update Saturation to eliminate background noise
	- Weighting wrt Hue color
- Arrow recognition
	- One detector is trained to recognize right arrow only. For left/straight arrows the image is rotated. 
	- The paper also incorporates prior information (right arrow under red light) into arrow recognition. --> but the paper did not see how.

#### Technical details
- Deceleration without discomfort to passengers is approximately **0.1 G**. Recognizing TFLs in the ranges exceeding **100 m** is required to make a natural intersection approach in automated driving.

#### Notes
- Questions and notes on how to improve/revise the current work  

