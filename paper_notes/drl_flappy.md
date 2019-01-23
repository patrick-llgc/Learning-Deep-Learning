# [Deep Reinforcement Learning for Flappy Bird](http://cs229.stanford.edu/proj2015/362_report.pdf)


_Jan 2019_

tl;dr: Trains a DQN to solve flappy bird.

#### Overall impression
It is quite impressive that the paper is a project paper from a stanford undergrad.  The project largely follows the DeepMind Nature 2015 paper on DQN. 

#### Key ideas
- Many of the ideas are standard from the Nature DQN paper
  - Multiple time-stamped states as input: the agent needs to learn velocity from a sequence of states. Input is therefore wxhxHistoryLength
  - Experience replay: decorrelates non-stationary and highly correlated experiences
  - Reward shaping: adding a rewardAlive to speed up training. The original reward is sparse, only incrementing when an obstacle is passed. 
  - $\epsilon$-greedy to solve exploration vs exploitation
  - Target network: Periodic updated  for training stability.
- DQN trained direclty on hard task cannot perform well on easy tasks. However, the author argues that train on easy tasks first an then fine-tune on hard tasks will solve this problem.
- The network is quit simple, with 3 conv layers with large kernel and large strides and 1 fc layer. 
- Training is not consistent. Training longer does not necessarily lead to better performance.

#### Technical details
- Background of the game (city skiline?) is removed to reduce clutter.
- Exploration probability changing from 1 to 0.1
- Reward discount=0.95
- rewardAlive=0.1, rewardPipe=1.0
- 600,000 iterations

#### Notes
- The network will probably perform equally well with the background as the background is stationary.
- The network architecture could be improved. 
- Tables IV and V shows the benefit of transfer learning. The tables are a bit hard to interpret at a glance. The columns represents the different networks, and the rows represents the deployment environment. Notably, in Table V, the DQN (hard) network performs really well in hard tasks but fails miserably in easy and medium tasks.
- The benefit from transfer learning seems very interesting, and contradicts the findings in supervised learning (such as noted in '[Overcoming catastrophic forgetting in neural networks](https://arxiv.org/pdf/1612.00796.pdf)'). The author say it only trained 199 iterations. Is this a typo?

#### Resources
- [github repo](https://github.com/yenchenlin/DeepLearningFlappyBird) for drl flappy bird

