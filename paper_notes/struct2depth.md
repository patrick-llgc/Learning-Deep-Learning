# [Struct2depth: Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos](https://arxiv.org/pdf/1811.06152.pdf) 

_July 2019_

tl;dr: Model the motion and size of each potential object to handle dynamic objects in the scene. 

#### Overall impression
The paper proposed several good ideas: 1) Model ego-motion and the motion of each dynamic object and warp the static scene with the dynamic objects together. 2) Online finetuning to improve performance. This comes at some cost, but is only possible as it uses self-supervision. This online finetuning method can be applied to other unsupervised method.

The improvement on prediction of depth in dynamic object is amazing. It also predicts the motion of each object! Also it tackles the infinite depth problem. Later efforts include [PackNet-SG](packnet_sg.md).

The paper's annotation is quite sloppy. I would perhaps need to read the code to understand better.

It directly inspired [depth in the wild](mono_depth_video_in_the_wild.md).

#### Key ideas
- Segment each dynamic object with Mask RCNN
- Regress the ego motion of the camera --> essentially visual odometry, Useful to compare with ORB-SLAM
- Regress motion for every dynamic object --> very nice intermediate result
- Combines warped and masked 
- To counter the infinite depth problem, introduce **prior size (estimated online)** and enforce consistency between estimated depth and calculated with prior.
- Online refinement method: takes in several frames, improves performance but at some computational cost at the beginning.


#### Technical details
- The depth of each object is roughly $D_{approx} \approx f_y \frac{p}{h}$. p is the prior size, and h is the height of the blob in pixels in perspective image. 

#### Notes
- Q: Why do they combine the mask of static scenes together to get over all static scene mask V?
- Q: based on the equation4, $V \cap O_i(S)=\emptyset$ so basically the recombined image should not have any occlusion at all.
- [Github code](https://github.com/tensorflow/models/tree/master/research/struct2depth)