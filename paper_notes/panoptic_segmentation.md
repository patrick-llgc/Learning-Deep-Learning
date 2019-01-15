# [Panoptic Segmentation](https://arxiv.org/pdf/1801.00868.pdf)

_Jan 2019_

tl;dr: A unified way to evaluate semantic segmentation and instance segmentation.

#### Overall impression
The paper proposed an intuitive way to address overlapped regions in semantic segmentation and instance segmentation. It also proposed Panoptic Quality (PQ) to evaluate panoptic segmentation.

#### Key ideas
* The paper proposed a new task named panoptic segmentation (PS) which unifies semantic segmentation and instance segmentation.
* PQ = SQ * RQ = average IoU of matched segments (masks) * F1 score of detetion.
* PQ is symmetric wrt the GT and prediction (since F1 and IoU are also symmetric)
* For small objects, human performance of RQ drops quickly, but SQ remain relatively stable. (if a small object is found, it is segmented relatively well.)
* PS can be done by performing semantic and instance segmentation indivisually and then combining them in two steps. First overlapping instance segments are resolved. Second, overlapping stuff and things segments are assigned to things. 

#### Notes
* In PS, each pixel of an image must be assigned a semantic label and an instance id. Object segmentation must be non-overlapping. This is different from instance segmentation.
* How to remove overlapped instances? The paper proposes to filter with two thresholds. First, instances with high enough scores (> threshold 1) are kept. Second, starting with the most confident instance, pixels are removed. If the remaining part is sufficiently large (>threshold 2) then it is kept. This question is of interest in medical imaging when lots of overlaps are not expected by end users. 
* There is a new type of task called "amodal segmentaton" which segments objects to its full extent, not only the visible part. 
