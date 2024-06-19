# [MPDM2: Multipolicy Decision-Making for Autonomous Driving via Changepoint-based Behavior Prediction](https://www.roboticsproceedings.org/rss11/p43.pdf)

_June 2024_

tl;dr: Improvement of MPDM in predicting the intention of other vehicles.

#### Overall impression
The majority is the same as the previous work [MPDM](mpdm.md). 

For the policy tree (or policy-conditioned scenario tree) building, we can see how the tree got built with more and more careful pruning process with improvements from different works.

* [MPDM](mpdm.md) iterates over all ego policies, and uses the most likely one policy given road structure and pose of vehicle.
* [MPDM2](mpdm2.md) iterates over all ego policies, and iterate over (a set of) possible policies of other agents predicted by a motion prediction model.
* [EUDM](eudm.md) itrates all ego policies, and then iterate over all possible policies of other agents to identify **critical scenarios** (CFB, conditioned filtered branching). [EPSILON](epsilon.md) used the same method.
* [MARC](marc.md) iterates all ego policies, iterates over a set of predicted policies of other agents, identifies **key agents** (and ignores other agents even in critical scenarios). 


![](https://pic3.zhimg.com/80/v2-a7778368cbf39f083ef5ad5a2f931a4e_1440w.webp)


#### Key ideas
- Motion prediction of other agents with a classical ML methods (Maximum likelihood estimation).

#### Technical details
- Summary of technical details, such as important training details, or bugs of previous benchmarks.

#### Notes
- Questions and notes on how to improve/revise the current work

