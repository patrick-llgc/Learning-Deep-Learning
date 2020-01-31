# [RRPN: Radar Region Proposal Network for Object Detection in Autonomous Vehicles](https://arxiv.org/abs/1905.00526)

_January 2020_

tl;dr: Use radar for region proposal in a two-stage object detector.

#### Overall impression
This study proposed translated anchors to address the misalignment of radar data.

One problem with radar data is the uncertain physical extent of radar pins. This include the lack of elevation z information, and the large uncertainty in azimuth direction.

However the method compared RRPN with Fast RCNN (even in 2019!). Selective Search may not be a fair comparison. The improvement of performance over Fast RCNN is not huge, and I suspect the performance may be lower than Faster RCNN.

#### Key ideas
- Radar detection s are not mapped to the center of the detected objects in every image. Therefore RRPN has differently translated anchors. Below is one example.
![](https://d3i71xaburhd42.cloudfront.net/f5420323c08d62e5c3265f3965dffa0d3edf1396/3-Figure1-1.png)
- Size of anchors are determined by 
$$S_i = \alpha \frac{1}{d_i} + \beta$$
	- the size of anchor is inversely proportional to the distance of the radar pin

#### Technical details
- Where do the Z info in the radar pins come from in Nuscenes?

#### Notes
- [code](https://github.com/mrnabati/RRPN)
- All interesting objects are collected by radar pins in Nuscenes. Is this true on other datasets as well?
![](https://mrnabati.github.io/publication/rrpn/featured_hua3fc4317165cb3b95af93b227b89972a_579309_680x500_fill_q90_lanczos_smart1_2.png)
![](https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_14%2Fproject_358521%2Fimages%2Fx9.png)