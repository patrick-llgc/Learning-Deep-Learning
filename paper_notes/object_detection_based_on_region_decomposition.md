# [Object Detection based on Region Decomposition and Assembly](https://arxiv.org/pdf/1901.08225.pdf)

@YuShen1116

_May 2019_ 

tl;dr: Summary of the main idea.
This paper propose a region decomposition and assembly(R-DAD) for more accurate object detection.
the motivation of decomposition object is to tackle occlusion: if your target object is 
blocked by something, decompose the object would help the network to learn the features from 
unblocked(valid) parts.

#### Overall impression
This paper is also working on the feature extraction part of the network to improve detection accuracy.
In the beginning they are still using the CNN network as backbone, and they applied a multi-scale region proposal and Roi
pooling to crate a base feature map. After that, they decompose the input object features into four parts(up, bottom, left, right) 
and use a region assembly block(RAB)  to extract and fuse features for those sub-images, the connect the block to the detection layer.


#### Key ideas
- Re-scale the feature from backbone to increase diversity of region proposals and filter output proposals wih low-confidence
and low-overlap ratios. (this step is more like a engineering trick)
- split the proposed object into four different parts, and re-scale them back with bi-linear interpolation. 

#### Technical details
- Multi-scale region proposal network
    - Input to this layer is the output of backbone
    - first use RPN to generate proposal boxes, then re-scales the boxes to increase the diversity or proposal regions(filter out the regions with low confident and low overlap to speed up)
- Region decomposition and assembly network
    - decompose the proposed region into different parts(top, bottom, left, right)
    - apply RoI pooling to the whole object region
    - up-sample each parts by a factor of 2 with bi-linear interpolation before feature extraction. 
    
#### Notes
- Questions and notes on how to improve/revise the current work
- a csdn note: https://blog.csdn.net/qq_30708445/article/details/88182603  
- This paper used lots of engineering tricks to achieve its performance, it's hard to say if his R-DAD network is really
working. However, I think decomposing the feature map for object detection is a very interesting approach. In the paper, 
the author only split the proposed region into four parts without considering their relationship. Maybe we can decompose 
the object with different methods? But this may requires the study of object relationship. Professor Tu had journal about image parsing
(http://pages.ucsd.edu/~ztu/publication/IJCV_parsing.pdf). Maybe we can figure something out from them
