# [Specialized Cyclist Detection Dataset: Challenging Real-World Computer Vision Dataset for Cyclist Detection Using a Monocular RGB Camera](https://drive.google.com/drive/u/0/folders/1inawrX9NVcchDQZepnBeJY4i9aAI5mg9)

_July 2020_

tl;dr: Cyclist dataset.

#### Overall impression
Very close to [Tsinghua Daimler Cyclists](tsinghua_daimler_cyclists.md), with more images, but fewer cyclist instances. But the part about pedestrians wearing special patterns make this paper almost like an informercial. 

#### Key ideas
- 60k images, 18k cyclist instances.
- Difficulty levels:
	- Easy: height > 40 pixel, fully visible
	- Medium: height > 25 pixel, less than 35% occluded
	- Hard: height > 25 pixel, and less than 60% occluded


#### Technical details
- BBD100k, cityscape and AutoNUE (IDD) datasets have separate bbox for bicycle and person, and no association information.

#### Notes
- Questions and notes on how to improve/revise the current work  

