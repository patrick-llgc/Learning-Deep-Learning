# [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://arxiv.org/abs/1711.02257)

_August 2019_

tl;dr: Dynamically adjust the weight of the tasks based on training progress.

#### Overall impression
[GradNorm](gradnorm.md) and [dynamic task prioritization](dtp.md) are very similar. However the weight rate adjustment is exponential as compared to focal loss (essentially also exponential), and the progress signal is loss drop as compared to KPI.

Task imbalance impede proper training because they manifest as imbalances between backpropogated gradients. The balance is struck when tasks are **training at similar rates as measured by the loss ratio L(t)/L(0)**.

Uncertainty weighting uses 1/L to adjust weight. --> this usually leads to weight too large too quickly, without normalization constraints. 

#### Key ideas
- The tedious grid search of hyperparameter can be efficient searched though a single parameter $\alpha$ to exponentially adjust the relative inverse training rate $r(t)$.
- Algorithm:
	- Pick the gradient last layer of the backbone to monitor
	- Get L2 norm of the feat for a task
	- Get task average of L2 norm for a particular time
	- The weights are not adjusted directly according to the progress but rather updated according to a GradNorm loss $Loss = \sum_{task}|G(t) - \bar{G}(t) \times [r(t)]^\alpha|$. the weights are renormalized to be 1 to ensure smooth training. 

#### Technical details
- using static weight to retrain the model from scratch. These values lie closely with the optimal grid-searched weights.
- The asymmetry parameter $\alpha$ controls how different the weight can be for diff tasks. The overall performance is relatively insensitive to the chance in $\alpha \in (0, 3)$. (0 means average weight).

#### Notes
- ipython notebook Demo on [github](https://github.com/hosseinshn/GradNorm/blob/master/GradNormv8.ipynb).

