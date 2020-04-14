# [PolarMask: Single Shot Instance Segmentation with Polar Representation](https://arxiv.org/abs/1909.13226)

_April 2020_

tl;dr: Extend [FCOS](fcos.md) for instance segmentation.

#### Overall impression
One main draw back of this paper is that the paper cannot handle scenarios where the center is not inside the mask, or there is no unique "center" inside the mask, such as a donut. 

One question is that given polar centerness metric, the center should always be inside the contour (other than donut shaped mask). This is an alternative to the distance transformation used by [ESESeg](https://arxiv.org/abs/1908.04067).

Overall the idea merits promotion, but the presentation of the idea is not impressive. I am still not convinced that the parameterization of the mask contour is optimal.

#### Key ideas
- Two heads: one predicting the centerness and the class, the other predicting the length from the rays emitting at constant intervals.
- Polar Centerness is a generalization of the Centerness score. Polar Centerness = $\sqrt{\frac{\min_i{d_i}}{\max_i{d_i}}}$. 
- Polar IoU loss: from the exact equation, the authors showed how to simplify it to a Polar IoU loss = $\frac{\sum_i(\min{d_i, d_i^*})}{ \sum_i(\max{d_i, d_i^*})}$ formulation, which is differentiable.

#### Technical details
- If a ray has multiple intersection points with the contour, choose the one with the max length. This will alter the topology of some masks. For example, a donut will become a circle.

#### Notes
- The idea is very similar to [ESESeg: Explicit Shape Encoding for Real-Time Instance Segmentation](https://arxiv.org/abs/1908.04067) <kbd>ICCV 2019</kbd>, however the ESESeg's performance is much worse as it tries to fit the boundary with special curve segments, while PolarMask regresses the mask contour derectly. 
- This work further inspired [FourierNet](https://arxiv.org/abs/2002.02709) which uses a differentiable shape encoder to represent the mask.
- Ideas for Improvements:
	- Predict at non-uniform intervals. Sample according to importance, similar to [PointRend](pointrend.md).
	- Predict multiple values per angle to address donuts.
	- Modify representation of contours. However the tricky part may be to come up with a differentiable loss approximate to the IoU loss. 