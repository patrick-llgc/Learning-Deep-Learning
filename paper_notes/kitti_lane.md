# [A New Performance Measure and Evaluation Benchmark for Road Detection Algorithms](http://www.cvlibs.net/publications/Fritsch2013ITSC.pdf) 

## [Monocular Road Segmentation using Slow Feature Analysis](https://www.honda-ri.de/pubs/pdf/1931.pdf)

## [Spatial Ray Features for Real-Time Ego-Lane Extraction](http://www.cvlibs.net/projects/autonomous_vision_survey/literature/Kuehnl2012IV.pdf)

_April 2019_

tl;dr: Proposed a new dataset for road detection and ego-lane detection.

#### Overall impression
The most interesting contribution of this paper is that it provided a broad review of SOTA evaluation metrics for road area and ego-lane detection.

#### Key ideas
- Two groups of metrics: those operate on the perspective image directly (**image representation**) and those operate on BEV image (**metric representation**).
- Segmentation based approach: pixel value evaluation, TP, FP on pixel level, IoU
- Lane lines/boundaries detection: distance between GT prediction markings. By allowing a flexible margin for counting successful border candidates, TP and FP can be defined. 
- In general, the metric evaluation results in BEV are always lower than that in the perspective space.
- **Evaluation in perspective space is biased and does not reflect the actual performance at regions far away adequately**.
- As long as you can tell which point (either in area or in the boundary), you can define TP, FP, precision, recall, F1, IoU, AP, and the precision-recall curve. 

#### Technical details
- For an ADAS warning a driver about oncoming narrow street sections, the width of the road needs to be measured in a distance of about 30-50m.
- When mapped to BEV, the resolution is 10 cm x 10 cm or 5 cm x 5 cm. The x and z range are [-10m, 10] x [~6m, ~50m].

#### Notes
- We need IPM (inverse perspective mapping) to convert perspective images to BEV.

