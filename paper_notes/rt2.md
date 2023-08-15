# [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://robotics-transformer2.github.io/assets/rt2.pdf)

_August 2023_

tl;dr: Web-scale pretraining via VLM improves generalization over RT-1. End-to-end trained vision-language-action (VLA) model to map robot observations to actions, while enjoying common sense reasoning of VLM.

#### Overall impression
VLM grounds the input of LLM with reality, and VLA grounds the output of LLM with reality. [VoxPoser](voxposer.md) is similar to [RT-2](rt2.md) in that it completes the grounding of both the input and output of LLM, but [RT-2](rt2.md) is better than VoxPoser in that this is end-to-end differentiable, and this is the **first model to complete data close-loop**.

[Say-can](saycan.md) and [PaLM-E](palm_e.md) only addresses the high-level task planning and still relies on **separate** low-level controllers to carry out action primitives ("skills"). They only take the end-to-end training to planning, which is hard to form data close-loop with field data without extensive human annotation.

[Gato](gato.md) designed new vision-language-action architecture from scratch and did not leverage the power of pretraining. The language part is only LM without explicit natural language training.

The usage of LLM has two benefits. First LLM/VLMs contain tremendous amount of common sense, and has a built-in "world-model" in it. Second the pretrained model has already amortized compute.

This builds on previous work of [RT1](rt1.md), but leverages the power of pretrained VLMs ([PaLM-E](palm_e.md), PALI-X) to improve the generalization over unseen cases (objects, backgrounds and environments). The performance over seen cases are roughly the same.

Similar work of integrating pretrained VLMs into end-to-end visuomotor manipulation policies, include CLIPort and MOO.

RT-2 direclty outputs action as special tokens. To avoid changing network architecture and uses pretraining as much as possible, it repurposes some tokens as action tokens (either individual number tokens or least used tokens in the vocab). This grounds LLM output beyond high level plans and into control action. --> This is quite similar to the idea of extending the natural language vocab in [Vision LLM](vision_llm.md). [Pix2seq v2](pix2seq_v2.md) does not use pretrained LLM and cannot output natural language tokens.


#### Key ideas
- VLA: vision-language-action models.
	- Express actions as text tokens and incorporate them into the training. 
	- This helps fit robotic actions and natural language responses into the same format. 
- Robot action 
	- 7 DoF (6 DoF pose + level of extention of gripper). Discretized into 256 bins.
	- episode termination token.
	- Reserving 256 tokens to serve as action tokens.
	- An action vector is converted into a string: "terminate, x, y, z, rotx, roty, rotz, gripper_extension"
	- Meaning of token is represented by the position. For example, 5 in x and 5 in y is not differentiated. --> This tokenization scheme of continuous variables is similar to and probably inspired by [Pix2seq](pix2seq.md).
- Base VLM: Two instantiations, with PaLI-X and [PaLM-E](palm_e.md).
- Training (fine-tuning)
	- The training uses co-finetune to finetune on original dataset and new robotics dataset (1/2 to 2/3), to avoid catestrophic forgetting. It is reported that this improves over finetuning with robot data only. --> This is actually bad news for industry application. Need a more modualarized design.
- Data: 
	- robot demo data: from RT-1. 13 robots over 17 months in an office/kitchen environment
- Inference
	- Outpuy constraint: only sampling valid action tokens when prompted with a robot-action task. It is also allowed to output natural lanauges when prompted with NLP tasks.
	- VLA only outputs actions for robot action tasks.
	- Realtime: deploy in the cloud and communicate via network. --> Need LLM acceleration such as AWQ
- Emergent capabilities
	- Symbol understanding: Move apple to 3
	- Reasoning: move apple to cup with the same color
	- Human recognition: bring code to Tylor Swift
- Limitations
	- RT-2 generalizes to semantics and visual concepts, it does not acquire any ability to perform new **motions**. The physical skills are still limited to the distribution of skills seen in the robot data. The dataset should vary more along the axes of skills. 
	- How to acquire new skills via watching humnas? This is one exciting future direction.

#### Technical details
- Robots requires grounded low-level actions, such as Cartesian end-effector commands. 
- Robot actions are treated as another language, cast into text tokens and are trained together with Internet-scale vision-language dataset (with upsampling of robotic dataset).
- Two types of VLM. Representation learning such as CLIP and BLIP, and language generation based on images. --> I should write a blog on [the topic of VLM](../topics/topic_vlm.md). 
- Language table benchmark tokenization scheme: Format of (X, Y), with X and Y taking integer values from -10 to 10, each represent 2D cartesian setpoints of the end effector.
- Better LLM leads to better performance in robotics.
- The alternative to using a pretrained LLM is to use pretrained visual-language representation via representation learning. (such as VC-1).


#### Notes
- The output is directly action. How can we tell if the action is reasonable or not? Maybe use the CoT idea to break into high level planning first, then do the action, to act as the bridge. --> The paper did not indicate if CoT can help with improving performance. Most likely yes, like [TidyBot](tidybot.md).
- Does not introduce action specific layers of the model, and recycles the natural language vocab space. --> Why not? May be a good practice when people do not have access to the large scale dataset to train VLMs.
- MOO: VLM is used as a separate module that enhances perception, but its representation are not used for policy learning.
- Chinese tech leaders Yao Lu and Tianhe Yu.


#### Questions
- Does RT-2 do high level task planning? If not, this may be an overkill. 
- How much computation was used in RT-2?
- How is PaLM-E better than Say-Can?
- World model (such as Daydreamer, [Gato](gato.md), [MILE](mile.md)) may benefit from LLM pretraining as well. 
