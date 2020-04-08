# [Multimodal Trajectory Predictions for Autonomous Driving using Deep Convolutional Networks](https://arxiv.org/abs/1809.10732)

_April 2020_

tl;dr: Multimodal behavioral prediction from Uber ATG with 6 seconds horizon.

#### Overall impression
Very similar to the idea of Waymo's [MultiPath](multipath.md). Uber's approach uses multiple trajectory prediction (MTP) loss. Waymo's approach uses fixed number of anchor trajectories. These two approaches are largely equivalent--predicting the mode first, and masking out the loss for all other modes. 

It uses a raster image to encode map information (BEV semantic map), very close to [MultiPath](multipath.md) and the previous researches such as RoR, ChauffeurNet and IntentNet.

It is quite interesting to see that a single modal model will just predicting the average of the two modes. In general, **if it is hard for humans to label deterministically, the underlying distribution is multimodal.**

![](https://cdn-images-1.medium.com/max/1440/1*utzVKtZOa_BDBL8GBd6XoA.png)

#### Key ideas
- Raster image as input
- (2 x H + 1) x M predictions per actor. 

#### Technical details
- Naively predicting Mixture of Experts loss $L^{ME} = \sum p_m L_m$ for m modes will leads to mode collapse. 
- Use multiple trajectory prediction loss. $L^{MTP} = -\sum I[m=m^*] \log p_m + \sum I[m=m^*] L_m $, where m* is the best matching prediction to GT.
-  Distance metric ADE does not seem to model multimodal behavior well. A better function measures distance by **considering an angle between the last
points of the two trajectories as seen from the actor position**, which improves handling of the intersection scenarios.
- The authors tried MDN (the one used by [MultiPath](multipath.md) from Waymo) and it leads to slightly worse performance then MTP. 

#### Notes
- **UKF is usually used for tracking object's trajectory. However it is single modal model. Its prediction at 1 s is reasonable, but at 6 s is not. **
- IRL is used for behavioral prediction. However usually it is not fast enough for real-time inference.

	> One approach would be to create a reward function that captures the desired behavior of a driver, like stopping at red lights, avoiding pedestrians, etc. However, this would require an exhaustive list of every behavior we’d want to consider, as well as a list of weights describing how important each behavior is. 
	
	> However, through IRL, the task is to take a set of human-generated driving data and extract an approximation of that human’s reward function for the task. Still, much of the information necessary for solving a problem is captured within the approximation of the true reward function. Once we have the right reward function, the problem is reduced to finding the right policy, and can be solved with standard reinforcement learning methods.
- MDN ([MultiPath](multipath.md) from Waymo used this formulation)

	>	Mixture Density Networks (MDNs) are conventional neural networks which solve multimodal regression tasks by learning parameters of a Gaussian mixture model. However, MDNs are often difficult to train in practice due to numerical instabilities when operating in high-dimensional spaces.


