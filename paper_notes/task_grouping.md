# [Which Tasks Should Be Learned Together in Multi-task Learning?](https://arxiv.org/abs/1905.07553)

_March 2020_

tl;dr: Answers this question: which tasks are better trained with others?

#### Overall impression
The paper uses the dataset from CVPR 2018 best paper [taskonomy](taskonomy.md), which studies the task transferability. Task grouping studies the multi-task learnability. The paper founds that they correlate inversely. 

The goal of multitask learning is two-fold.

- find best performance (with the regularization power from training with other tasks)
- reduce inference time

One of the key insights from the paper is: 

> The inclusion of an additional task in a network can potentially improve the accuracy that can be achieved on the existing tasks, even though the performance of the added task might be poor.

#### Key ideas
- Optimal grouping is better than single multi-task network, or multiple single-task network. 
	- For example, the best strategy found by this paper is train 2.5 networks, 2 full-sized networks with 2 tasks each and the third half-sized network train the fifth task. However the fifth task is needed to regularize the first two full-sized network to gain optimal performance for the first four tasks. 
- Given enough computation resource, training individual single networks are better, but sometimes need other tasks to help with regularization
- **Task grouping (multitask learning) is inversely correlated with task transferability.** Thus it is better to train dissimilar tasks together. --> This is somewhat counter-intuitive. The authors argue that this will give more meaningful regularization.
- The paper proposed two methods to reduce computation burden
	- Early stopping: validation score at 0.2 epoch already correlates with final score pretty well. This saves 20x computation resource. 
	- High order approximation: train all single and dual task models and use them to approximate higher order grouping. This reduces computation from exponential combination to quadratic.

#### Technical details
- Hard parameter sharing: same backbone/encoder
- Soft parameter sharing: Same architecture and L2 distance penalty between corresponding weights. Or added peephole connection between corresponding weights. --> This does not improve inferenece time. 

#### Notes
- Questions and notes on how to improve/revise the current work  

