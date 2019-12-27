# [Deep Active Learning for Efficient Training of a LiDAR 3D Object Detector](https://arxiv.org/abs/1901.10609)

_December 2019_

tl;dr: Use active learning to reduce the amount of labeled data.

#### Overall impression
In conventional deep learning, data are labeled in a random fashion, and they are fed into a training pipeline with random shuffling.

In active learning, a model iteratively evaluates the informativeness of unlabeled data, selects the most informative samples to be labeled by human annotators and updates the training sets.

The detection task is a simplified version of classification and size/depth regression inside frustum.

Active learning is better than random baseline, regardless of the uncertainty evaluation method.

#### Key ideas
- Active learning also alleviates data imbalance problem as it actively queries images that are most informative rather than their natural frequency.
- This paper focuses on the uncertainty of classification. It improves the data efficiency for cls by 60%, but only a minor increase in localization task. --> This could be improved by introducing localization uncertainty into active learning framework.
- MC-dropout and deep ensemble (training multiple models with the same architecture but diff init.) is better than single softmax output --> **the difference is quite small, so softmax is good enough in reality**

#### Technical details
- One way to evaluate quality of predictive uncertainty is **sparsification plot**. A well-estimated predictive uncertainty should correlate with the true error. And by gradually removing the predictions with high uncertainty, the average error over the rest of the predictions will decrease. 
	- If correct, the sparsification plot should be monotonically decreasing.


#### Notes
- Lidar/image labeling tool: label 2D box in images, and human annotators only needs to label lidar points in frustum.
- [Localization-Aware Active Learning for Object Detection](https://arxiv.org/abs/1801.05124) <kbd>ACCV 2018</kbd>

