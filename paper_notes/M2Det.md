# [M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network](https://arxiv.org/abs/1811.04533)

@YuShen1116

_May 2019_ 

tl;dr: Summary of the main idea.
A new method to produce feature map for object detection.

#### Overall impression
Describe the overall impression of the paper. 
Previous methods of building feature pyramid(SSD, FPN, STDN) still have some limitations because 
their pyramids are built on classification backbone. This paper states a new method of generating 
feature pyramid, and integrated into SSD architecture. 
As a result, they achieves AP of 41.0 at speed of 11.8 FPS with single-scale inference strategy on 
MS-COCO dataset. 

#### Key ideas
- Use Feature Fusion Modules(add figure later) to fuse the shallow and deep features(such as conv4_3 and conv5_3 of VGG) 
from backbone.
- stack several Thinned U-shape Module and Feature Fusion Module together, 
to generate feature maps in different scale(from shallow to deep). 
- Use a scale-wise feature aggregation module to generate a multi-level feature pyramid from above features.
- Apply detection layer on this pyramid. 


#### Notes
- Not very easy to train, it costs 3 - more than 10+ days to train the whole pipeline.   
- I think this idea is interesting because it states that the features from classification backbone is not good enough
for object detection. Modifying the features for specific task could be a good direction to try. 
