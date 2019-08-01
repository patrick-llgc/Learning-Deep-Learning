# [Mono3D: Monocular 3D Object Detection for Autonomous Driving](https://www.cs.toronto.edu/~urtasun/publications/chen_etal_cvpr16.pdf)

_August 2019_

tl;dr: The pioneering paper on monocular 3dod, with tons of hand crafted feature.

#### Overall impression
The paper is quite innovative at the time being, but looks rather archaic three years later in 2019. In particular, it chose to use Fast (not Faster) RCNN as base net, together with tons of hand crafted features. Many of the cited work are pre-DL era. From industry's standpoint of view, this paper is of little practical use as of 2019.

#### Key ideas
- Proposal generation by placing dense proposals (~14K) with prior templates on several proposed ground plane, then score. Heavily influenced by the line of work of faster RCNN. The proposal generation part has the following criterion as scoring function.
	- Generate location heatmap as prior for proposal generation.
	- Cross a patch below the car bbox as context. 
	- Semantic: if the projection of 3d bbox proposal falls under the semantic map.
	- Shape: manually crafted feature (counting how many contour pixels are in one of the 3x3 cells)
- The top candidates are then further classified and regressed using Fast RCNN. One branch has original ROI pooling, the other having ROI pooling from enlarged bbox. This is quite similar but not the same as the shared feature map in [monoPSR](monopsr.md). The feature map regresses orientation offset as additional task on top of Fast RCNN (orientation anchor are only two at 0 and 90 degrees).

#### Technical details
- The paper largely uses fast RCNN structure to regress the 

#### Notes
- Questions and notes on how to improve/revise the current work  

