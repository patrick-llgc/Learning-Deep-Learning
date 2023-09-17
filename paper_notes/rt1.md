# [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)

_August 2023_

tl;dr: A generalist robotic control agent, powered by a data-absorbent model architecture, trained on diverse and large-scale robotics dataset. 

#### Overall impression
The key to foundation models and LLMs lies in the open-ended, task agnostic training. (GBB is deadend.) RT-1 aimed to be the LLM for robotic control. RT-1 input is language and vision observations, and maps them to robot actions. It learns robot policies to solve language conditioned tasks from vision.

Robotic dataset is more expensive to collect (requires engineering-heavy automation, or human demo), and thus training a multitask backbone model is of importance. --> **Synthetic data** is perhaps another way?

RT-1 focuses on low-level control tasks (skills), such as picking up an apple. RT-1 is not able to do high level task planning. In say-can terminology, RT-1 focuses on the can part, not the say part. This is perhaps why RT-1 does not use pretrained LLM/VLM model, and [RT-2](rt2.md) may be an overkill. (Based on the assumption that RT-2 cannot do high level task planning.) RT-1 does not use pretrained LLM and thus is quite similar to [Gato](gato.md) and [Mile](mile.md), and the world-model papers (such as daydreamer and dreamer series).

RT-1 exhibits good performance and generalization capability, and it can perform 97% seen tasks and 75% unseen tasks. Much better generalization capability than [Gato](gato.md). High capacity model enables generalization. Transformer is such an architecture. Real time (RT) performance also requires efficient architecture. Pun intended? (In contrast, RT-2 is quite heavy and not that RT.)

Data is more important than model architecture, and data diversity is more important than data quantity. Breadth > Scale. Tasks should be well connected as well.

The main contribution is the large dataset. But how effective is this dataset when the end effector is changed? Perhaps not that useful.



#### Key ideas
- Dataset
	- 130k episodes of 700 tasks, collected with 13 robots over 17 months. 
	- Training is collected in robot classroom, modeled after Kitchen 1 (K1). 
	- Testing is on two kitches, K1 and K2. 
	- 8 skills (verb), 700 tasks (instructions, verb + noun)
- Model architecture
	- Transformer: seq2seq. Inputs (language conditioned images) are mapped to a sequence, and output actions are also mapped to a sequence. 
	- Language conditioning via FiLM (a way to construct VLM, cf [Flamingo](flamingo.md). 一种通过语言嵌入来调节视觉活动的经典技术)
	- Image: no patchify, flattened to 9x9 tokens, reduced to 8 tokens by TokenLearner. 6 images yields 48 tokens. This is the input to the transformer. 
	- BC (behavior cloning) and capped by human demo
- Action formulation 13 DoF
	- 7 DoF arm
	- 3 DoF base movement
	- 3 DoF modes switch: arm, base or termination flag.
- Action Tokenization
	- 256 bins, for each variable, the target is mapped into one of the 256 bins.
	- Changing this to continuous variable will drop performance significantly (30%+ drop). 
- Inference
	- 3 Hz.
	- Autoregressive generation of actions does not help. Actions are not generated autoregressively, to speed up inference. 
- Evaluation and key takeaways
	- Simulation helps, a lot.
	- Data from other robots of diff morphology helps. However skills cannot be transfered directly, without finetuning with data with the target morphology.
	- RT-1 generalizes to new background and objects much better than [Gato](gato.md). This means RT-1 can support long horizon tasks, as many as 50 steps. 
	- Data diversity > data quantity. Smaller dataset is more useful than a narrower dataset. 

#### Technical details
- HW
	- Everyday Robots (EDR) as the main platform. 7 DoF arm, twp-fingered gripper, model base. 
	- Kuka robots IIWA
- An episode: the full interaction i (instructions), {x_j, a_j} from the starting step to terminating step is referred to as an episode. A binary reward indicating the success of the execution is given at the end of an episode. 
- Several dimension to evaluate model: performance, generalization, robustness.
- Mobile manipulation: navigation and manipulation


#### Notes
- As a product, the model needs to be continuously improved via data close-loop. Consider Mid-journey, Feishu, and Nycle. **A small but iterable model is better than a large, non-iterable model.**
- How does saycan with RT-1 compare with RT-2? and PaLM-E?

