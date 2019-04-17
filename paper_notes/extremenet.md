# [ExtremeNet: Bottom-up Object Detection by Grouping Extreme and Center Points](https://arxiv.org/pdf/1901.08043.pdf)

_April 2019_

tl;dr: Detect four extreme corners and the center of an object with an anchor-less framework. This is heavily influenced by [CornetNet](cornernet.md).

#### Overall impression
The bottom up 

#### Key ideas
- Top-down vs bottom-up:
	- Faster RCNN series all uses the concept of "region". This is the **top down** approach.
	- CornerNet, ExtremeNet and CSP detects the center, corners or extreme points of an object. This is the **bottom up** approach.
	- Most objects are not axis-aligned boxes, and fitting bbox includes many distracting background pixels.
- ExtremeNet predicts 5 heatmaps, each of the four extreme points, and one for center point. 
- The grouping method is purely geometric. All combinations of the extreme points are enumerated and check the calculated center against the center heatmap. 
- Instance segmentation: The extreme points (or the generated octagon) are fed into DEXTR (deep extreme cut) for predicting a mask, which performs competitively with Mask RCNN. 
- Error analysis shows that the center heatmap is trained quite well, but the extreme points heatmap need improvement.
- **Ablation test and error analysis**
	- Previous papers usually have extensive ablation test, but not so much error analysis. In error analysis, the GT of each factor is fed into the pipeline to see the boost on the performance. If the performance boost is significant, that identifies the bottleneck of the pipeline.

#### Technical details
- A corner lies outside an object, but extreme points lie on the object with visual evidence. 
- Corners vs Extreme points
	- Corner points can be retrospectively extracted from bbox, but extreme points can only be retrospectively extracted from instance masks. 
	- However if annotated from scratch, extreme points are faster to obtain than corners.
	- Extreme points are not unique if the object is axis-aligned box. ExtremeNet uses the center point of an axis aligned edge. 
- Gaussian maps in the context of keypoint regression. The standard deviation is either fixed or proportional to the object size. The Gaussian kernel can be used as the *regression target* in the L2 loss case or as the *weight map* to reduce penalty near the one-hot positive position in a logistic regression case (cross entropy case).
- Ghost box suppression (this perhaps could also be useful for CornetNet), Edge aggregation (to resolve ambiguity of extreme points).

#### Notes
- Questions and notes on how to improve/revise the current work  

