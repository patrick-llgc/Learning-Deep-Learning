# [SUPER: A Novel Lane Detection System](https://arxiv.org/abs/2005.07277)

_June 2020_

tl;dr: Slope compensated lane line detection.

#### Overall impression
Nothing too impressive about this approach. The approach is not even end to end differentiable and uses a nonlinear optimizer for solution. This is not quite transferrable. 

It only targets to solve 90% of the problem (parallel lane lines) and still does not solve split or merge issues.

#### Key ideas
- A novel loss that involves entropy and histogram. The main idea is that in BEV space the lane line points collapsed to the x dimension should have multiple equally spaced peaks. But this loss is not differentiable.
- Approximate a road slope. This is essentially the pitch estimation of the road in [LaneNet](lanenet.md).

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

