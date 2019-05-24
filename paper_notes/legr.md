# [LeGR: Filter Pruning via Learned Global Ranking](https://arxiv.org/pdf/1904.12368.pdf)

_May 2019_

tl;dr: Learn a global ranking metric (affine transformed L2 norm) for layers so pruning filters is just as easy as thresholding this measure. This paper provides a practical tool to prune filters and outperforms prior SOTA.

#### Overall impression
Previous methods use costly methods to find the Pareto front of performance-resource tradeoff curve. **Resource-Constrained filter pruning** is the main topic of this study, and it performs better than Auto Model Compression (AMC) and MorphNet. Global filter ranking can also be achieved by first order Taylor approximation to the loss increase caused by pruning (sample data needed thus not a data-free method)


#### Key ideas
- NAS is a bottom up approach, but more resource demanding. Top-down approach, such as pruning or quantization are more practical.
- Trivial sparsity is easy to achieve just by setting a L1 or L2 norm of all the weights, but it usually only leads to model size reduction but not computation efficiency. Achieving structured/non-trivial sparsity is more valuable. 
- Overall workflow:
	- Calculate L2 norm
	- Transform L2 norm with affine transformation pair
	- Filter with global thresholding
- LeGR uses regularized evolution algorithms by picking the fittest model and mutate them through one-step random walk in a selected subset of layers. The random walk changes the affine parameter estimation.
- LeGR can search once and applying different threshold to prune network to different level, but for both AMC and MorphNet has to finetune for each constraint level. LeGR is more efficient for deployment.
- Finetuning 50 steps is good enough to evaluate the $\alpha$-$\kappa$ pair (the best model). 
- **The pretrained model does not need to converge in order to be pruned.**

#### Technical details
- Under the assumption that the loss function is [Lipschitz continous](https://en.wikipedia.org/wiki/Lipschitz_continuity) with a finite derivative upper limit, then $\min (L' - L)$ becomes $ \min \sum_i (a ||\Theta_i|| + b) h_i + c, \text{ s.t., } C(1-h) \le \sum_i d h_i + e$, whose Lagrangian form is 
	$$\min \sum_i (\alpha_{l(i)} ||\Theta_i|| + \kappa_{l(i)}) h_i = \min \sum_i M_i h_i$$
where $h_i$ is the deletion mask, l(i) is the layer of i-th filter, $||\Theta_i||$ is the norm of the i-th filter, and M_i is the learned metric of i-th filter. This minimization problem can be achieved by simple global thresholding. 
- LeGR is 5-7x faster than AMC/MorphNet in searching (finetuning the same amount of time), and overall 2-3X speedup in pruning pipeline.


#### Notes
- What did the authors use? DDPG or regularized evolution algorithm?
- LeGR can search once and applying different threshold to prune network to different level, which makes it easy to deploy.
- Code available on [Github](https://github.com/cmu-enyac/LeGR).
