# [Depth from Videos in the Wild: Unsupervised Monocular Depth Learning from Unknown Cameras](https://arxiv.org/pdf/1904.04998.pdf)

_July 2019_

tl;dr: Estimate the intrinsics in addition to the extrinsics of the camera from any video.

#### Overall impression
Same authors for [Struct2Depth](struct2depth.md). This work eliminates the assumption of the availability of intrinsics. This opens up a whole lot possibilities to learn from a wide range of videos. 

This network regresses depth, ego-motion, object motion and camera intrinsics from mono videos. Thus it is named learn-K (intrinsics) --> The idea of regressing intrinsics is similar to [GLNet](glnet.md).

#### Key ideas
- Estimate each of the intrinsics
- Occlusion aware loss (picking the most foreground pixels during photometric loss calculation)
- Foreground mask to mask out the possible moving objects. 
- Use a randomized layer optimization (this is quite weird)

#### Technical details
- Sometimes an overall supervision signal is given to two tightly coupled parameters and it is not enough to get accurate estimate for both parameters. (cf. [Deep3Dbox](deep3dbox.md))

#### Notes
- In detail, how was the lens correction regressed?
- See interview with the CEO of isee [on this paper](https://medium.com/syncedreview/google-ai-unsupervised-depth-estimation-for-arbitrary-videos-51d97ec0d70).
- Q: Can we project the intermediate representation (3D points) to BEV instead of back to camera plane for loss calculation? This would eliminate the need for using occlusion-aware loss. 
- [code](https://github.com/google-research/google-research/tree/master/depth_from_video_in_the_wild) is based on that of [struct2depth](struct2depth.md).