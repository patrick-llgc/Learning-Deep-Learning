# [3D-LaneNet: End-to-End 3D Multiple Lane Detection](http://openaccess.thecvf.com/content_ICCV_2019/papers/Garnett_3D-LaneNet_End-to-End_3D_Multiple_Lane_Detection_ICCV_2019_paper.pdf)

_March 2020_

tl;dr: First paper on Monocular 3D Lane line detection. 

#### Overall impression
3D LaneNet is the first work on 3D lane line detection. This is very close to [Tesla's 3D lane line detection](https://twitter.com/theteslashow/status/1223049982191685633?lang=en). 

3D LaneNet does not need fragile assumptions such as the flat ground assumption, only assumes zero camera roll wrt to the local road surface. The network **estimates the camera height and pitch** (together with known camera intrinsics, they are all that are needed to determine the homography between BEV and perspective).

The network architecture is a dual-pathway backbone which ctranslates between image and BEV space. This is similar to [Qualcomm's deep radar detector](radar_fft_qcom.md). However the transformation parameter is estimated on the fly by a localization network, similar to [Sfm-learner](sfm_learner.md), which is essentially a special case of [Spatial Transformer Network](stn.md). This is another way to lift features to BEV than [Orthogonal Feature Transform](oft.md).

The system also works for 2D lane line detection, and reaches near SOTA performance on TuSimple dataset. The regression of different lanes at preset longitudinal (y location in perspective image) is widely used in industry.

#### Key ideas
- Dataset 
	- The study used simulated environment (Blender) to generate large amount of synthetic data with 3D groundtruth. 
	- They also collected 2.5 hrs of real data with lidar detector and semi-manual annotation, but only to validate the idea. --> This is not quite scalable. Maybe synthetic dataset and sim2real is the way to go. (cf 3D Synthetic LLD dataset [Gen-LaneNet](gen_lanenet.md))
- Image to BEV with predicted pitch $\phi_{cam}$ and height $h_{cam}$. Then project the image according to the homography determined by these two parameters. (We assume zero camera roll).
- The 3D lane line can be formulated as the projection of the 3D point onto the $P_{road}$, the plane tangent to the local road surface, and then the z-distance to the road surface. 




#### Technical details
- Why IPM? Lanes are ordinarily parallel in this view and their curvature can be accurately fitted with low-order polynomials. Lane markings are similar regardless of the distance from camera.
- Anchor-based regression
	- K predefined y position (6 points). Each of the K points predicts two numbers, dx and z. Each anchor also has one conf predicting if there is a lane associated with the anchor at a predefined position $Y_{ref}$ = 20m. 
	- Each anchor is 16 cm (first layer) x 8 = 128 cm wide. Anchor number N = W/8. 
	- Total target is N * (2K + 1) per lane type. It can predict 3 types (2 center-lines for fork/merge and one deliminator).
- The dual pathway backbone is quite heavy. To get better trade-off, late-stage IPM provides second best performance.

#### Notes
- [LLD paper summary](https://github.com/amusi/awesome-lane-detection): updated quite frequently.
- [Review on 知乎](https://zhuanlan.zhihu.com/p/113165034)