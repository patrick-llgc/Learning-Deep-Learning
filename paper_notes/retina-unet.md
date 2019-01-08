# [Retina U-Net: Embarrassingly Simple Exploitation of Segmentation Supervision for Medical Object Detection](https://arxiv.org/pdf/1811.08661.pdf)

_Jan 2019_

tl;dr: The addition of full segmentation supervision at the last decoding layer to FPN boosts the performance of object detection.

#### Overall impression
This paper proposed an interesting idea to combine U-Net with FPN. This idea could be further extended by introducing deep supervision on every level of FPN, instead of just the last decoding layer at P0.

#### Key ideas
* Additional loss terms (pixel-wise cross entropy and soft Dice loss) are added to the classification and bbox losses. 
* The segmentation loss term is added to P_0, the original image scale. The detection loss is still performed on P2 to P5, and does not add to the computational cost of object detection. 
* Weighted box clustering (WBC) aggregates multiple box results from TTA. It leads to superior results than NMS based on IoU as demonstrated by the authors. There are three main ideas of WBC.
  - Both the scores and the coordinates are averaged. Conventional NMS only keeps the largest scored bbox with its original score.
  - The bboxes are weighted by score and a confidence factor $w = f \cdot a \cdot p$. f is the overlap with the highest scored bbox, a is the area (empirical) and p is distance from the patch center. 
  - The aggregated score is down-weighted by the total number of views or voters in TTA or ensemble. 

#### Notes
* The paper also implemented 3D Mask RCNN as a strong baseline. This itself should be a huge contribution.
* The addition of full segmentation supervision can be done to both one-stage (Retina U-Net) and two-stage (U-Faster RCNN) object detection pipeline. The addition of segmentation loss is on the image level, not per region level.
* The object detection from P2 to P5 is the same as the original FPN paper. Note thta RetinaNet used P3 to P7 to reduce computational cost and to accomodate the detection of large objects. 

