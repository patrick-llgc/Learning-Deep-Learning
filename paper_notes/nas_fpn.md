# [NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection](https://arxiv.org/pdf/1904.07392.pdf)

_May 2019_

tl;dr: Using NAS to search for best cross-scale connection in FPN.

#### Overall impression
First paper to use NAS to search optimal pyramid architecture for object detection. NAS-FPN finds scalable design which better benefits from repeating the modules. However it requires quite a lot of computational resource to search and not easily reproducible in small labs. Most manly engineered cross-scale connections are shallow (only connecting neighboring or the same scales) but NAS-FPN can connect non-neighboring scales as well.

#### Key ideas
- Stacking vanilla FPN's multiple times does not always leads to improvement (degrading after repeating 3 times).
- [PANet](panet.md) uses an extra bottom-up pathway to improve OD performance. 
- Modular design which has identical input and output feature levels. 
- Merging cell merges any two input feature layers (could be from different scales but must have same number of filters) into an output layer. Each merging cell has four selections to make
	- Input feature layer 1
	- Input feature layer 2
	- Output feature resolution
	- Merging binary op (either sum or global pooling [This global pooling is different from the pooling op that squashes all spatial content]). The scaling is achieved by bilinear upsampling or maxpooling.
- Anytime prediction: deeply supervised during training, but can stop at any layer depending on the application platform **dynamically**.
- Different tradeoffs:
	- Stacking FPNs
	- Using different backbone
	- Adjusting feature dimension (channels of feat maps)

#### Technical details
- Cosine learning schedule helps stabilizing training process.
- The number of architectures converges after 8000 searches. The best architecture appears at 8000 step and multiple times after that. 
- NAS finds that high resolution features need to be connected directly to the high resolution output layers (i.e., it never scaled down and then up).
- Dropblock improves the performance dramatically. Here is a [summary from 知乎](https://www.zhihu.com/question/300940578) of how to drop different features effectively. 

#### Notes
- Q: How about searching 14-cell NAS-FPN instead of stacking two 7-cell FPN?
- Q: why there could be 3 inputs in generating a new layer, given that sum and global-pool are binary operators?
- Q: How to guarantee that 7-cell finds features in all 5 scales? 
	- A: For a 7-cell process, last 5 cells predict the five output scales, but the order is searched and predicted automatically (not predefined).
- Q: why the anytime prediction (deeply supervised) performs worse?
- Good review from [知乎](https://www.zhihu.com/question/320662763) on [NAS-FPN](https://zhuanlan.zhihu.com/p/63932921) and [FPN optimization](https://zhuanlan.zhihu.com/p/63047557).
- SSDLite from [MobieNetV2](mobilenets_v2.md) needs another read. 
