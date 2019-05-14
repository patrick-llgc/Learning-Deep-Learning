# [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)

_May 2019_

tl;dr: The author states the lottery ticket hypothesis: A randomly-initialized, dense neural network contains a 
sub-network that is initialized such that -- when trained in isolation -- it can match the est accuracy of the original 
network after training for at most the same number of iterations. This could be a good approach to compress the NN 
without harming performance too much.

#### Overall impression
By using the pruning method of this paper, the winning tickets(pruned network) are 10-20% (or less) of the size of the 
original network. Down to that size, those networks meet or exceed the original network's test accuracy in at most the 
same number of iterations. 

#### Key ideas
- Summaries of the key ideas
- Usual pruning method
    - Randomly initialize a neural network.
    - Train the network for j iterations, arriving at parameters \theta_j.	
    - Prune p% of the parameters from \theta_j.
    - Reset the remaining parameters and re-train the pruned-model(winning ticket)
- Paper's pruning method
    - Above pruning is one-shot, the author focus on iterative pruning, 
    which repeatedly trains, prunes, and resets the network over n rounds.
    - each round prunes p^(1/n)% of the parameters.
    - Other steps are the same
    - Re-initialization after pruning would destroy the performance. 
    
#### Technical details
- DO NOT re-initialize the model after pruning.
- This pruning method is sensitive to learning rate. It requires warm up(increase learning rate by step) to find winning tickets
at higher learning rates.

#### Notes
- It would be a good approach to try this iterative prune method. 
However, this paper is applying a complex model(VGG, Resnet) on simple dataset(cifar-10, mnist), 
it's really hard to say the real performance of the pruning. 
- The reason for warm up is still un-certain
