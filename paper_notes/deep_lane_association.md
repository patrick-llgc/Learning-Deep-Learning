# [Deep Metadata Fusion for Traffic Light to Lane Assignment](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8613841)

_August 2019_

tl;dr: Convert metadata to binary masks and use deep metadata fusion to improve traffic light to lane assignment.

#### Overall impression
This work improves upon the previous [only-vision baseline](https://ieeexplore.ieee.org/document/8569421) (from same authors, without any meta data).

The authors converted all meta data to binary masks, as all the meta data are spatial info. --> This method may not work well if some meta data cannot be spatially encoded. 

Embedding meta data info into conv:

- [meta data fusion for TL2LA](deep_lane_association.md)
- [fusing radar pins with camera](distant_object_radar.md)
- [cam conv](cam_conv.md) to fuse focal length into convs.
- [camera radar fusion net](crf_net.md)


The traffic lights to lane assignment problem can be solved traditionally using huristics rule-based methods, but not very reliably. Only-vision approach (enhanced with enlarged ROIs), and with meta data, works almost as good as human performance in subjective test. This is because the relevant traffic lights are not always mounted directly above the their associated lane.

**I feel that lane localization is very challenging.** How do we know if the current lane is right turn only? Are we going to rely on HD maps or vision alone for this? At least some vision based methods should be used. 

#### Key ideas
- **Rule based methods** achieves about 60%-80% precision in different (complex/full) driving scenarios. Rule-based methods largely works in non-complex cross-sections
	- largest and nearest traffic light in driving direction is relevant
	- it has to be the traffic light with the largest size and highest position from the group with most traffic lights of the same color
	- a laterally mounted traffic light between the left and right ego-vehicle lane line markings is relevant. If no traffic light is within the ego-vehicle lane line markings, the traffic light with the shortest lateral distance tot he ego-vehicle lane center will be selected as relevant.
- Deep fusion methods perform largely as well as concat method. 
- Heterogeneous meta data: spatially encode to the same size as input map.

#### Technical details
- Architecture:
	- Input 256x256x3 + metadata feature map (MFM)
	- Output: 256 x 3 for ego, left and right lanes, indicating position of all relevant column positions. 
	- Loss: regression loss.
- Input image: bottom half is IPM (inverse perspective mapping) data. Same as previous only-vision baseline.
- Deep fusion: Convert location of detections/meta data to binary attention maps and element-wise multiplied with first F=12 layers. This proves to be slightly better than directly concat the MFM.
- Evaluation: 
	- Subjective test: a typical driver would have about 7 seconds for the last 100 m up to an intersection.
	- Images with only one lane are excepted.
	- Accuracy evaluated wrt distance bins

#### Notes
- IPM has two basic assumption: the road is flat and the road is free of obstacles. The second is an extension of the first one. When there is obstacle on the ground such as vehicle, there will be bleeding artifact.
![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/868aaca19e48a6299900172ef2e11e940cd30435/2-Figure1-1.png)
![](https://ars.els-cdn.com/content/image/1-s2.0-S1566253514001031-gr19.jpg)

