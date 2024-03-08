# [RPT: Robot Learning with Sensorimotor Pre-training](https://arxiv.org/abs/2306.10007)

_March 2024_

tl;dr: Masked modeling to pretrain a multimodal robotic foundation model.

#### Overall impression
Pretraining outperforms training from scratch, transfers across tasks, labs and robots, and scales well.

This paper focusesd on masked modeling, instead of autoregressive/generative modeling. The paper also tried causal masking but still with noncausal attention, and the single task results seems about the same. This is an incomplete trial of autoregressive pretraining. --> Overall it seems that the industry is moving to generative modeling for better scalability and more available recipes. It would be nice to see a generative model variant. 

Robotic data contains rich sensory and motor information (action) difficult to capture with visual pretraining alone. The unlabeled sensorimotor trajectories implicitly encode the structure of the physical world and we can use them to learn sensorimotor representaitons for downstream robotic tasks. The paper uses a masking strategy of all modalities and time with a high (70%-90%) masking ratio. This particular masking strategy is critical in encouraging the model to learn a cross-modal, spatio-temporal representations.

Note that the sensorimotor trajectory in this paper also includes images now, different from [Locomotion as next token prediction](locomotion_next_token_pred.md) which excludes images.

#### Key ideas
- Architecture
	- Vision encoder: pretrained from MVP, a prev work from the same group, which leverates internet scale video for pretraining.
		- Compared to prediciton in pixel space, prediction in the latent space makes the task more tractable. --> Echoed by LeCun.
	- Seprarate modality encoders. Thus no need for modality embeddings. All modality encoders use linear layer to project to 192-d.
	- Positional embedding to represent time, shared across modalities. 
	- Transformer with bidirectional self-attention.
- Training 
	- Pretraining: MSE loss in latent space.
	- Downstream transfer: finetuning all layers is significantly better than linear probing with frozen model.
- Masking strategy
	- Pretrainig: random masking with large masking ratio. Caucal masking (next token prediction) yields similar results per 
	- Finetuning: only masking action at the last time step. This finetune step enables autoregressive prediction.

#### Technical details
- Sensorimotor trajectories: camera images, proprioceptive states, and actions. These trajectories are generated with classic methods (motion planning + grasping algorithm).
- Task Difficulty: Pick < Destack < Stack.
- HW
	- 7 DoF Franka + 1 DoF gripper, control at 10Hz.
	- 1 Egocentric cam + 2 Exocentric cams.
- **Data quality matters a lot.** Includes failure cases in pretraining will significantly ruin the performnace. 
- Emergent self-correction: if the robot fails to grasp an object initially, it would move back and proceed to grasp the object successfully.

#### Notes
- This paper wins an oral, as it involves real robotic experiments. It seems that a lot of the robotics papers focusing only in simulation does not get good reviews. 
- Predicts actions for the next 16 steps. --> How is this rollout achieved? Not mentioned in the paper. This could provide a good handle to safeguard the autoregressively generated actions. 
