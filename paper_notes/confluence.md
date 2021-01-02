# [Confluence: A Robust Non-IoU Alternative to Non-Maxima Suppression in Object Detection](https://arxiv.org/abs/2012.00257)

_December 2020_

tl;dr: Confluence is a confidence weighted Manhattan distance inspired proximity measure. It is an alternative to NMS in post-processing and requires No change the neural network.

#### Overall impression
There largely exists two methods of NMS, IoU based NMS (as in RetinaNet and FCOS) and center location based (as in [CenterNet](centernet.md)). Confluence uses the heavy cluster of bounding boxes as an indicator of the presence of an object. 

Confluence does not select the best bbox by cls conf scores, nor it uses IoU to suppress others. NMS suffers from a hard-coded arbitrary IoU threshold. Center-distance based NMS such as the maxpooling used in [CenterNet](centernet.md) also has this issue. 

It is more robust to highly occluded scenes. Gains in recall are much higher. 

Unfortunately the paper is poorly written, with confusing notations and abuse of terminology (proximity, confluence). The pseudo-code is not helping at all. We really need to wait for the github implementation to understand the details. For example, the WP is not even mentioned in the pseudo-code. And how confluence is updated with more proximity is not clear.

#### Key ideas
- Confluence embraces the tendency for dense object detectors to return numerous bbox within the RoI. Confluence finds clusters of bboxes. --> This is caused by the many-to-one label assignment and lack of classification cost in assignment. See [DeFCN](defcn.md) and [OneNet](onenet.md).
- Two steps:
	- retaining: pick box with lowest (best) confluence score.
	- removal: bbox within close proximity (with a thresh) of the bbox.

#### Technical details
- My understanding of the pseudo-code:
	- For each pair, calculate the Manhattan Distance. This is called **Proximity**.
	- For a given box, calculate weighted average of all proximity < 2, weighted by cls conf score. This is called **Confluence**. 
	- Find lowest confluence score, retain it, and remove all other box with proximity < threshold.
	- Repeat the 3rd step.
- The illustration in Fig. 4 and 5 are just for visualization of the concept. The proximity is calculated wrt a random bbox. This is not used in the algorithm.
![](https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgmsKU8gXsrWA913B6d5Oh6xkz3lZ1yuWoEHiaIYnZIrPubmd9rBJbgXuNZTCNKXU5wBxWLQy3rOUjQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

#### Notes
- [Review on 知乎](https://mp.weixin.qq.com/s/snLxpvUAWphO3xzfPOpdCQ)
- Both confluence score and proximity score is the lower the better. This is a bit confusing. It would be better to define confluence score in a way that it is the higher the better, and the higher the lower proximity score is. 
