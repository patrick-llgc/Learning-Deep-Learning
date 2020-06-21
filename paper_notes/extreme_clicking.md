# [Extreme clicking for efficient object annotation](https://arxiv.org/abs/1708.02750)

_December 2019_

tl;dr: Annotate extreme points to replace conventional bbox annotation leads to same accuracy, reduced annotation time and additional information.

#### Overall impression
This paper inspired [ExtremeNet](extremenet.md). 

This work is extended to [DEXTR: Deep Extreme Cut: From Extreme Points to Object Segmentation](https://arxiv.org/abs/1711.09081) that extends extreme points to instance segmentation masks.

#### Key ideas
- Conventional bbox annotation involves clicking on imaginary corners of a right box around the object. This is difficult as these corners are often outside the actual object and several adjustments are required to obtain a tight box.
- Annotation by extreme point clicking is only **7s per object instance, 5x faster** than the traditional way of drawing bbox.
	- Extreme points are not imaginary but well defined points on the object
	- No separate box adjustment step is required. 
- Add a qualification test. Find extreme points, find all pixels with x or y within 10 pixels of extreme values, include all pixels within 10 pixels of any of the selected pixels. All these pixels are acceptable. --> we may need to adjust these thresholds for smaller objects. 

#### Technical details
- Two ways to obtain annotation: "annotation party" vs crowdsourcing. The former is too costly and crowdsourcing is essential for creating large datasets.
- The bbox annotator need to pay attention to extreme points anyway to ensure accurate annotation. Clicking the top-left corner couples the localization and aligning hairlines of of top and left-most extreme point at the same time.
- Using grabCut to automatically find masks. These masks can train CNN that is 1.5 mAP below that trained with full mask.

#### Notes
- This is promising to help annotating on distorted and undistorted image simultaneously.
- Can we train a patch-based segmentation model to help with this task?
- Maybe we can use siam-mask to try auto annotation on videos.

