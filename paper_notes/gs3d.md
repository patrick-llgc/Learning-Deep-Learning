# [GS3D: An Efficient 3D Object Detection Framework for Autonomous Driving](https://arxiv.org/abs/1903.10955)

_October 2019_

tl;dr: Get 3D bbox proposal (guidance) from 2D bbox + prior knowledge, then refine 3D bbox. 

#### Overall impression
This paper also regresses 2D bbox and orientation with conventional 2DOD architecture, then get a coarse 3D position, then refine. The approach of generating initial 3D location is similar to [FQNet](fqnet.md) and [MonoPSR](monopsr.md).

The depth estimation method is practical. The quality aware loss is also easy to implement than IoU net to predict quality of bboxes. However the usefulness of surface feature extraction is doubtful. 

The paper still uses Caffe in 2019 is a bit of a shocker. 


#### Key ideas
- Guidance generation: This step only uses 2D bbox and intrinsics to get the 3D location of the cuboid center.
	- **Depth estimation** is based on a strong prior of bbox height. Based on the dataset, the 3D top center is the top of 2D bbox, the 3D bottom center is close to the bottom of 2D bbox (for KITTI, 0.07 above). Essentially, the distance is estimated by the ratio of 93% of bbox and subtype average height. --> this is really hacky. Maybe using the width of the car is a better constraint?
	- The idea of using keypoint to estimate depth can also be found in [MonoGRNet from the Russian team](monogrnet_russian.md).
- Architecture: 
	- 2DOD + orientation
	- surface feature extraction fused with RoI Align for 3D property refinement


#### Technical details
- **Representation ambiguity**: Using RoI aligned feature to regress 3D offset, as the authors argues, has representation ambiguity. The authors used surface feature extraction from **projective RoIAlign** from 3 visible surfaces. --> In retrospect, this perhaps will work with projecting the 3D wireframe to 2D image [FQNet](fqnet.md).
- 3D refinement used multibin loss proposed by [deep3dbox](deep3dbox.md) to replace direct regression. Bin width is the stdev of the error based on training dataset. Sigmoid is used instead of softmax to handle classification of BG cases. 

- Same as always, Global yaw = ray (azimuth) + observation angle (local yaw)
- **Quality aware loss**: in 2D detection, the target's label is changed to reflect IoU quality. This is simpler implementation than [IoU Net](iou_net.md).

#### Notes
- Questions and notes on how to improve/revise the current work  

