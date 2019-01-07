# [Retina U-Net: Embarrassingly Simple Exploitation of Segmentation Supervision for Medical Object Detection](https://arxiv.org/pdf/1811.08661.pdf)

_Jan 2019_

tl;dr: The addition of full segmentation supervision to FPN boosts the performance of RetinaNet.

#### Key ideas
* Additional loss terms (pixel-wise cross entropy and soft Dice loss) are added to the classification and bbox losses. 
* The segmentation loss term is added to P_0, the original image scale. The detection loss is still performed on P2 to P5, and does not add to the computational cost of object detection. 
* Weighted box clustering aggregates multiple box results from TTA. It leads to superior results than NMS based on IoU as demonstrated by the authors.

#### Notes/Questions
* The addition of full segmentation supervision can be done to both one-stage (Retina U-Net) and two-stage (U-Faster RCNN) object detection pipeline. The addition of segmentation loss is on the image level, not per region level.
* The object detection from P2 to P5 is the same as the original FPN paper. Note thta RetinaNet used P3 to P7 to reduce computational cost and to accomodate the detection of large objects. 

#### Overall impression
==Fill this out last; it should be a distilled, accessible description of your high-level thoughts on the paper.==