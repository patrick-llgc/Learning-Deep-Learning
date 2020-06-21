# [Vehicle Re-ID for Surround-view Camera System](https://drive.google.com/file/d/1e6y8wtHAricaEHS9CpasSGOx0aAxCGib/view)

_June 2020_

tl;dr: Vehicle reID with fisheye surround cameras. 

#### Overall impression
Another practical work from Zongmu. The tracking

#### Key ideas
- single camera reID
	- This should be largely tracking. --> why this is called ReID?
	- SiamRPN++ model for single camera tracking
- multi camera reID
	- BDBnet model (batch dropblock, [ICCV 2019](https://arxiv.org/abs/1811.07130), SOTA for pedestrian ReID)
	![](https://img-blog.csdnimg.cn/2020030610390223.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4Mjg0OTYx,size_16,color_FFFFFF,t_70)
	- association rule based on physics and geometry to only perform vehicle ReID in overlapping FoV
	- feature distance and geometry constraint distance
	![](https://cdn-images-1.medium.com/max/1280/1*QDFk2SdahCa0xI4zlUv2iQ.png)

#### Technical details

#### Notes
- [Talk at CVPR 2020](https://youtu.be/WRH7N_GxgjE?t=2570)
- I feel that only the physical constraint will already do very good cross camera reID of the same vehicle. Do we really need reID module?

