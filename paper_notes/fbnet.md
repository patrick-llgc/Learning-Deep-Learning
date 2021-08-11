# [FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search](https://arxiv.org/abs/1812.03443)

_August 2021_

tl;dr: Gradient-based optimization (Differentiable NAS) with hardware aware loss function.

#### Overall impression
FBNet (facebook-berkeley net) is one of the representative papers of efficient NAS. 

The paper is similar in idea to, but easier to understand than [DARTS](darts.md). The main idea is to train a **stochastic super net** with probabilistic combination of all possible building blocks, then sample the network during inference. 

The hardware aware optimization via a LUT is inspired by [MnasNet](mnasnet.md), which uses evolution algorithm as a controller. 

The differentiable NAS method resembles pruning a lot.

#### Key ideas
- The search space is represented by a stochastic super net. Each layer has 9 parallel blocks each controlled by a probability. 
	- The descrete sampling process is not differentiable, and we have to use the [Gumbel trick](https://towardsdatascience.com/what-is-gumbel-softmax-7f6d9cdcb90e) to make the sampling process differentiable.
- Training procedure: the weights w and the architecture distribution parameters a are trained in an alternating fashion.
	- For each epoch, 80% of the images are used to train w.
	- 20% of the images are used to train a.
- A hardware specific LUT makes the latency differentiable wrt layer-wise block choices.
	- The design of LUT assumes that the runtime of each operator is independent of other operators. 
	- $L(a, w) = CE(a, w) \times \alpha \log(LAT(a))^\beta$. LAT is a latency on the target device. 
- The macro-architecture is fixed and the search space is layer-wise.

#### Technical details
- The optimality of a CNN architecture is conditioned on many factors such as **input resolution** and **target device**. Once these factors change the optimal architecture is likely to change. 
	- On a different device the same op can have diff latency. 
	- Computational cost is quadratic to input size.
- The network is sampled multiple times during inference. 
- RL-based NAS is time consuming as the controller needs thousands of architectures to train the controller. 
	- Return is cumulative reward in RL.

#### Notes
- The method of RNN + RL to do NAS is long gone. DARTS dominates the SOTA now. See [Youtube tutorial](https://www.youtube.com/watch?v=AmitvRzmvv0).
	- The main issue was that the objective function (validation accuracy is not a differentiable function of the controller RNN parameters). So RL is used to handle the sparse reward. This is a main difference between gradient descent (requires a differentiable objective) and RL.
	- RL essentially learns a cost function if it is hard to design.
- [A great review of Gumbel trick](https://towardsdatascience.com/what-is-gumbel-softmax-7f6d9cdcb90e). It has one reparameterization trick, refactoring the sampling of Z into a deterministic function of the parameters and some independent noise with a fixed distribution. Then replacing the argmax with softmax controlled by a temperature.
- [Youtube tutorial in Chinese on DARTS](https://www.youtube.com/watch?v=D9m9-CXw_HY)

