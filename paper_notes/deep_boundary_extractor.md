# [BoundaryExtractor: Convolutional Recurrent Network for Road Boundary Extraction](http://openaccess.thecvf.com/content_CVPR_2019/html/Liang_Convolutional_Recurrent_Network_for_Road_Boundary_Extraction_CVPR_2019_paper.html)

_August 2020_

tl;dr: Extract road boundary with polylines. 

#### Overall impression
There are several works from Uber ATG that extracts polyline representation based on BEV maps.

- [Crosswalk Extractor](deep_structured_crosswalk.md)
- [Boundary Extractor](boundary_extractor.md)
- [Polyline Loss](hran.md): lane lines
- [DAGMapper](dagmapper.md): merges/forks

This work goes from the structured inference pipeline to an autoregressive **convolutional snake (cSnake)** pipeline. This is further extended to purely end-to-end RNN-based design in [Polyline Loss](hran.md) and [DAGMapper](dagmapper.md).

#### Key ideas
- Input: lidar + BEV + lidar height grad (sobel, then take magnitude), 5 ch
- Output (deep features):
	- Detection map (Inverse distance transform, DT, 1 ch)
	- End points of boundary (usually at patch boundary, 1 ch)
	- Direction map (normal direction towards the closest road boundary, 2 ch). --> This is better than the baseline of alignment angle (dilated normal) in [Deep Structured Crosswalk](deep_structured_crosswalk.md).
- Inference with cSnake. cSnake iteratively attends to rotated ROIs and outputs the vertices of a polyline corresponding to a road boundary.
	- Predict endpoint first
	- Based on endpoint, crop and rotate patch of concatenated detection and direction map for prediction of next point.
	- Do this autoregressively. --> this is similar to RNN used in [RoadTracer](road_tracer.md), [Polyline Loss](hran.md) and [DAGMapper](dagmapper.md). Note that all methods cropps an ROI before feeding into the conv-LSTM module. 
- **Amortized learning**: The cSnake can be trained using either the ground truth or predicted deep features. For our best model, we train using half ground truth and half predicted deep features. This is better than training with predicted deep features alone. --> this can be used as a trick to train two-stage systems.

#### Technical details
- 90% Recall/Precision/F1. 99.3% topology accu.
- 4 cm/pixel resolution
- Distance transforms a natural way to “blur” feature locations geometrically ([source](https://www.cs.cornell.edu/courses/cs664/2008sp/handouts/cs664-7-dtrans.pdf)). This is another way to densify sparse GT as compared to Gaussian blurring as in [CenterNet](centernet.md).
- **Skeletonized DT pred is a strong baseline**.
- Scoring of each polyline: average of detection score for each point. This score is used for filtering of low score FPs and polyline NMS.
- Evaluation of connectivity. 1 GT should have 1 pred, not fragmented pred. This is measured as CDF. 

#### Notes
- Removal of dynamic objects with semantic segmentation. No need for instance segmentation.
- GT and evaluation assignment with Hausdorff distance. --> why not Chamfer distance?
- Chamfer distance vs Hausdorff distance. [Tutorial here](https://www.cs.cornell.edu/courses/cs664/2008sp/handouts/cs664-8-binary-matching.pdf) and [here](https://courses.cs.washington.edu/courses/cse577/04sp/notes/dtrans-tutorial-2004.pdf).
