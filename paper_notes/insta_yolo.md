# [INSTA-YOLO: Real-Time Instance Segmentation](https://arxiv.org/abs/2102.06777)

_March 2021_

tl;dr: Extend yolo to perform single-stage instance segmentation.

#### Overall impression
Insta-yolo adopts a fixed length contour representation, and uses a 

Work by Valeo Egypt. Speed is very fast but quality is subpar. Looks like a run-of-the-mill paper.

#### Key ideas
- Represent masks by a fixed number of contour points (polygons) in Cartesian, and predict the polygons of each object instance through the center of the object. 
- GT generation with a deterministic algorithm (dominant points detection).
- Loss
	- Regression loss wrt the GT generated with deterministic algo
	- IoU Loss to compensate for the fact that no unique representation for the object mask using fixed number of vertices.
- This can also work for orientated bbox prediction.

#### Technical details
- Log Cosh loss: a differentiable alternative to Huber loss (smooth L1 loss).

#### Notes
- [On the detection of dominant points on digital curve](https://www.researchgate.net/publication/3191687_On_the_detection_of_dominant_points_on_digital_curve)

