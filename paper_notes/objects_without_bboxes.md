# [Locating Objects Without Bounding Boxes](https://arxiv.org/abs/1806.07564)

_December 2020_

tl;dr: Locating objects with center points, but with a new loss function based on average Hausdorff distance.

#### Overall impression
This paper proposed a NMS-free method for object detection. The method by predicting a heat map is very similar to that of [CenterNet](centernet.md) and other anchor-free methods. However it proposes a new set-prediction loss based on the averaged Hausdorff distance. 

Sometimes bbox of an object is not the optimal representation. Object localization is more appropriate where objects are very small, or substantially overlap. The bbox in crowded scenes may be infeasible to be used as GT. 

#### Key ideas
- [Chamfer distance](hran.md) or [Earth Mover's distance](crowd_det.md) are closely related to and very similar to Hausdorff distance.
- Hausdorff function is highly sensitive to outliers, ergo the average Hausdorff distance.
$d_{AH}(X, Y) = \frac{1}{|X|} \sum_{x \in X} \min_{y \in Y} d(x, y) + \frac{1}{|Y|} \sum_{x \in Y} \min_{y \in X} d(x, y)$
- Average Hausdorff distance operates on coordinates, not heatmap. To use heatmap directly with the loss, a weighted Hausdorff loss is proposed. 


#### Technical details
- [Generalized mean function](https://en.wikipedia.org/wiki/Generalized_mean) 
$M_n[f(a)] = (\frac{1}{|A|} \sum_{a \in A} f(a)^n)^{1/n}$
	- When n --> inf, then it is max function
	- When n --> -inf, then it is min function
	- When n = 1, it is mean function.
- Using a generalized mean will make min more smooth.

#### Notes
- I wonder how this will compare with the [centerNet](centernet.md) based keypoint approach.

