# [Astyx camera radar: Deep Learning Based 3D Object Detection for Automotive Radar and Camera](https://www.astyx.net/fileadmin/redakteur/dokumente/Deep_Learning_Based_3D_Object_Detection_for_Automotive_Radar_and_Camera.PDF) 

_January 2020_

tl;dr: Camera + radar fusion based on AVOD.

#### Overall impression
 

#### Key ideas
The architecture is largely based on [AVOD](avod.md). It converts radar into height and intensity maps and uses the pseudo image and camera image for region proposal.

#### Technical details
- Bbox encoding has 10 dim (4 pts + 2 z-values) in the original AVOD paper. However this paper said it used 14 dim. 
- Radar+camera does not detect perpendicular cars well. However it detects cars that align with the direction of the ego car much better. 


