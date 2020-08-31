# [RAR-Net: Reinforced Axial Refinement Network for Monocular 3D Object Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2822_ECCV_2020_paper.php)

_August 2020_

tl;dr: Use deep reinforcement learning to refine mono3D results.

#### Overall impression
The proposed [RAR-Net](rarnet.md) is a plug-and-play refinement module and can be used with any mono3D pipeline. This paper comes from the same authors as [FQNet](fqnet.md). Instead of passively scoring densely generated 3D proposals, [RAR-Net](rarnet.md) uses an DRL agent to actively refine the coarse prediction. Similarly, [shift RCNN](shift_rcnn.md) actively learns to regress the differene.

[RAR-Net](rarnet.md) encodes the 3D results as a **2D rendering with color coding**. The idea is very similar to that of [FQNet](fqnet.md) which encodes the 2D projection of 3D bbox as a wireframe and directly rendered on top of the input patch. This is the "direct projection" baseline in [RAR-Net](rarnet.md). Instead, [RAR-Net](rarnet.md) uses a parameter aware data enhancement. and encodes semantic meaning of the surfaces as well (each surface of the box is painted in a specific color).

The idea of training a DRL agent to do object detection or refinement is not new. It is very similar to the idea of [Active Object Localization with Deep Reinforcement Learning](https://arxiv.org/abs/1511.06015) <kbd>ICCV 2015</kbd>.

![](https://bardofcodes.github.io/DRL_in_CV_Papers/Papers/img/A1-1.png)


#### Key ideas
- The curse of sampling in the 3D space. The probability to generate a good sample is lower than in 2D space. 
- **Single action**: moving in one direction at a time is the most efficient, as the training data collected in this way is the most concentrated, instead of scattered throughout 3D space. 
- One stage vs Two stage vs MDP
	- one stage is not good enough. Two stages are hard to train separately. Thus MDP via DRL, as there is no optimal path to supervise the agent. 
- [DQN](nature_dqn_paper.md)
	- input: cropped image patch + parameter-aware data enhancement (color coded cuboid projection)
	- output: Q-values for 15 actions. (2 * 7 DoF adjustment + one STOP/None action)
	- Each action is discrete during each iteration, and it is in allocentric coordinate system instead of global system. This helps to learn the DQN same action for the same appearance. 
	- Reward is +1 if the 3D IoU increase, -1 if decreases.


#### Technical details
- The training data is generated from jittering the GT. 
- There are **two commonly used split for KITTI validation dataset, one from mono3D from UToronto, and one from Savarese from Stanford**. It should be checked to perform apple-to-apple comparison. Similarly, there are different validation split for monocular depth estimation.

#### Notes
- What is the alternative to RL? There is one baseline with all the features as RAR-Net but without RL in Table 5.

