# [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115) 

_August 2019_

tl;dr: Self-paced learning based on homoscedastic uncertainty. 

#### Overall impression
The paper spent much math details deriving the formulation of mutitask loss function based on the idea of maximizing the Gaussian likelihood with himoscedastic uncertainty. However once implemented, the formulation is extremely straightforward and easy to implement. 

#### Key ideas
- Uncertainties (for details refer to [uncertainty in bayesian DL](uncertainty_bdl.md))
	- Epistemic uncertainty: model uncertainty
	- Aleatoric uncertainty: data uncertainty
		- Data dependent (heteroscedastic) uncertainty
		- Task dependent (homoscedastic) uncertainty: does not depend on input data. It stays constant for all data but varies between tasks. 
- Modify each loss by uncertainty factor, $\sigma$. 
$$L \rightarrow \frac{1}{\sigma^2}L + \log\sigma $$
This formulation can be easily generalized to almost any loss function. There is a task-specific parameter that can be learned and dynamically updated throughout learning.
- Instance segmentation is done in a way very similar to center net. Each 

#### Technical details
- Regress $\log \sigma^2$ instead of $\sigma^2$ directly. This exponential mapping allows to regress unbounded scalar values. 

#### Notes
- OPTICS clustering algorithm (Ordering points to identify the clustering structure) is similar to DBSCAN, but less sensitive to parameter settings. See tutorial [here](https://pro.arcgis.com/en/pro-app/tool-reference/spatial-statistics/how-density-based-clustering-works.htm) and [coursera video](https://www.coursera.org/lecture/cluster-analysis/5-3-optics-ordering-points-to-identify-clustering-structure-JiYeI).

