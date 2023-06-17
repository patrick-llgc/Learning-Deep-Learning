# [UniSim: A Neural Closed-Loop Sensor Simulator](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_UniSim_A_Neural_Closed-Loop_Sensor_Simulator_CVPR_2023_paper.pdf)

_June 2023_

tl;dr: Differentiable neural sensor simulator for camera and lidar.

#### Overall impression
This work presents a neural sensor simulator for both camera and lidar. Many neural networks are carefully orchestrated to generate photorealistic sensor data. UniSim is able to handle extrapolated views better, by incorporating learnbale priors for dynamic objects, and leveraging a CNN to complete unseen regions. 

The goal is to generate **editable and controllable digital twin**. 

Generating "what-if" scnearios (perturbed scenes) from a single recording would be a game changer for developing self-driving cars. **Photorealistic simulation allows closed-loop evaluation in safety-critical scenarios without the safety hazard**. 

The introduction is very well written and articulated the importance of photorealistic simulation for closed-loop testing of autonomous driving system. The results seem to be a huge progress compared with previous SOTA, but still leaves some gap for industrial application. Many technical details are not clearly presented either. 

#### Key ideas
- Separation of dynamic objects and static objects. --> This is very similar to the autolabel pipeline for occupancy networks, presented in [SurroundOcc](surroundocc.md) and [OpenOccupancy](openoccupancy.md).
- Novel view synthesis
	- Method 1: gemeotry reconstruction and then warp pixel features into the new view.
	- Method 2: represent scene implicitly as neural radiance field (Nerf) and perform rendering with neural network.
- Domain gap for perception
	- Sim2Real and Real2Sim results are very similar to Sim2Sim for log replay. (The logged scene is reconstructed with UniSim with the original trajectories) Still some gaps remain for extrapolated (lane shift) cases. --> It would be interesting to see examples of the gap, but the paper did not report.

#### Technical details
- Neural feature field (NFF): superset of occupancy networks and Nerf. 
- UniSim has a special handling of distant regions such as the sky, to extend background scene to unbounded scenes. 
- Photometric loss, perception loss, and a discriminator loss are used to ensure photorealism.


#### Notes
- For deployment for all technology, evaluation is the key, and should be the handle for technical management.
- Tech companies needs to secure business scenarios.
- TODO
	- Sample based planner used in this paper: [Jointly learnable behavior and trajectory planning for self-driving]()
	- Similar work to this also from Raquel's team: [CADSim]() <kbd>CoRL 2022</kbd>
