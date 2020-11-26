# [Traffic light recognition exploiting map and localization at every stage](https://web.yonsei.ac.kr/jksuhr/papers/Traffic%20light%20recognition%20exploiting%20map%20and%20localization%20at%20every%20stage.pdf)

_November 2020_

tl;dr: Very thorough description of using HD map for TFL recognition.

#### Overall impression
Although the perception models the paper uses is quite outdated, it has a very clear discussion regarding how to use HD maps online.

Also refer to [TFL map building with lidar](tfl_lidar_map_building.md) for a similar discussion.

#### Key ideas
- Prior maps (with lat/long/height of TFLs) improves accuracy of recognition and reduces algorithm complexity. 
	- **Task trigger**: Recognition algorithms do not have to operate continuously as perception begins only when the distance tot he facing TLF is within a certain threshold
	- **ROI extraction**: this limits the search area in an image
	- Estimate the size of a TL
- Procedure
	- RoI extraction with safety margin. Slanted slope compensation. Road pitch needs to be stored in the HD map as well.
	- Detection locates TFL in image
	- Classify state of TFL
	- Tracking estimate position of TFL. Threshold for association should adjust based on distance.

#### Technical details
- The effect of pitch (on a bumpy road) is bigger for TFL at long distances. On average the pitch change could be up to +/- 2 deg.

#### Notes
- Questions and notes on how to improve/revise the current work  

