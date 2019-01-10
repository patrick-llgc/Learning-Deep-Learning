# [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

_01/06/2019_

tl;dr: the founding paper of DQN

#### Key ideas

* Approximating action values (Q) with neural nets are known to be unstable. Two tricks are used to solve this: experience replay buffer, and a periodically updated target network.
* The authors tied the important ideas of adjusting representation based on reward (end-to-end learning) and replay buffer (hippocampus) with biological evidence. 

#### Notes/Questions

* Drawbacks: It does not make much progress toward solving Montezuma's revenge.

> Nevertheless, games demanding more temporally extended planning strategies still constitute a major challenge for all existing agents including DQN (for example, Montezuma’s Revenge).

Overall impression: fill this out last; it should be a distilled, accessible description of your high-level thoughts on the paper.
