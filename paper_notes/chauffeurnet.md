# [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/pdf/1812.03079.pdf)

_April 2020_

tl;dr: Imitation learning (behavior cloning) for motion planning by synthesizing corner cases.

#### Overall impression
Typical building blocks of autonomous driving and other robotics system include perception (including sensor fusion), prediction (behavior prediction of other cars), planning (of motion of ego car). The rest of the engineering stack (such as motion control to get steering and braking) is more of mechanical engineering. 

Motion planning can be trained with reinforcement learning (RL) or imitation learning (IL) or conventional motion planning. The difference between IL and RL is the IL uses offline data alone and RL is online learning (need to simulate the environment). 

ChauffeurNet takes in the results from perception and directly outputs the planned trajectory. Behavior prediction for other agents is downplayed and is just one auxiliary task. This is not exactly end-to-end but is more flexible and scalable and can leverage the power of synthetic data. (cf [Gen-LaneNet](gen_lanenet.md))

Motion planning's hardest part is the need for **closed-loop tests**, which means prediction is feed into the loop for planning the next step. It test the systems's ability to self-correct and recover from adverse scenario. **Open-loop tests** means prediction/planning of next step but without feeding into the loop of next state. The paper showed that a model that performs well on open-loop tests do not necessarily perform well on close-loop test. 

In imitation learning, or behavioral cloning, when naively applying supervised learning (which assumed iid input) to MDP, there will be a distributional drift, as the action from the last step may affect observations in the next state. The typical way to address the compounding error/distributional drift issue in imitation learning is **[Data Aggregation (DAgger)](https://arxiv.org/abs/1011.0686)**. (DAgger has humans in the loop and asks expert to label data when observed from following learned policy in closed-loop test. This way you get some supervision on how to correct mistakes.) The perturbation method in ChauffeurNet is essentially addressing issues with synthetic Dagger pipeline. 



#### Key ideas
- Architecture
	- Input: BEV road layout and perception results as bbox. Speed limit and traffic light info are also encoded into the BEV image. 
	- Output: trajectory of ego car (agent): (waypoint p, heading $\theta$, speed s).
	- Includes Feature Net (backbone extracting features) and AgentRNN (LSTM conv head to predict waypoint and bbox). LSTM has fixed manually crafted memory and not learned memory, and has a fixed time window and trained in an unrolled fashion. 
- Losses
	- Imitation loss of waypoint and bbox
	- Environmental loss: collision loss, on road loss, geometry loss (car must follow expert's rut or virtual rail). 
	- Auxiliary loss: Predict on-off road regions (binary segmentation) and **behavior prediction loss** of all other cars in the scene. 
- Use intermediate input and output representation makes it easier to do closed loop test with simulation before testing on real cars. 
- Perturbing the expert path by displacing one point and smoothing the whole path. Start with the perturbed point as the first point (see Fig. 9). This is essentially a **virtual DAgger** pipeline.

#### Technical details
- Field of interest: BEV 800x800 image, 0.2 m per pixel.
- By "agent" the paper means ego car, by "perception boxes" the paper means other cars. 
- History: 1 second, 5 frames with 0.2 interval
- Past motion dropout: as the past motion history during training from expert demo, the net can learn to cheat by extrapolating from the past rather than finding the underlying causes of the behavior. In closed-loop testing this breaks down because the past history is from the net's own past predictions. With dropout proba of 0.5, we keep only the current position of the agent.
- Imitation dropout: only use environmental loss to guide agent's learning. This is better than simply down-weigh the imitation loss. 
- Open-loop test: unlike closed-loop test, the predictions are not used to drive the agent forward and thus the network never sees its won prediction as input. 

#### Notes
- [ICML 2020 workshop](https://slideslive.com/38917927/chauffeurnet-learning-to-drive-by-imitating-the-best-and-synthesizing-the-worst)
- [CS234](https://youtu.be/V7CY68zH6ps?t=1789) on imitation learning and DAgger.
- Waymo can use their drivers data directly (with possible minor filtering) as most of their drivers are expert (thus "chauffeurs"). Tesla and other OEM may have a harder time harvesting high quality driving segments and thus require some filtering.
- Not sure what the route planner does.
- [Driving Policy Transfer via Modularity and Abstraction](https://arxiv.org/abs/1804.09364) <kbd>CoRL 2018</kbd> trains policy with the output from a segmentation network, thereby enabling transfer learning to the real world using a different segmentation network trained on real data. This is very similar to cf [Gen-LaneNet](gen_lanenet.md). However we may need to do some trick to bridge the gap between real results from the last stage and the perfect GT we use to train the next stage. 