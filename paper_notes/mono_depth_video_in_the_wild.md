# [Depth from Videos in the Wild: Unsupervised Monocular Depth Learning from Unknown Cameras](https://arxiv.org/pdf/1904.04998.pdf)

_July 2019_

tl;dr: Estimate the intrinsics in addition to the extrinsics of the camera from any video.

#### Overall impression
This work eliminates the assumption of the availability of intrinsics. This opens up a whole lot possibilities to learn from a wide range of videos. 

This network regresses depth, ego-motion, object motion and camera intrinsics from mono videos.

#### Key ideas
- Estimate each of the intrinsics
- Occlusion aware loss (picking the 
- Foreground mask to mask out the possible moving objects. 
- Use a randomized layer optimization (this is quite weird)

#### Technical details
- Summary of technical details

#### Notes
- In detail, how was the lens correction regressed?