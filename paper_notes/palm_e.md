# [PaLM-E: An embodied multimodal language model](https://arxiv.org/abs/2303.03378)

_July 2023_

tl;dr: Creation of embodied multimodal LLM through injecting continuous embodied observations (images, state estimates) into the language embedding space of a pretrained LLM.

#### Overall impression
This paper is similar to [Gato](gato.md) in that they both target to become generalist agent. Yet there are two distinct differences between the two.

- [Gato](gato.md) does not use a pretrained LLM. [PaLM-E](palm_e.md) injects multimodal input into LLM and aims to leverage the internalized knowledge ("common sense")
- Gato directly output control signal, while PaLM-E cannot output control signal, and PaLM-E still uses natural language as the interface to chain with low-level control tasks. 

**How to combine these two seems to rely on how to translate natural language to low level control, or to ground LLM not only on the input side, but also on the output side.** 

PaLM-E transfers knowledge from visual-language domain into embodied reasoning (robot planning). PaLM-E operats on multimodal sentences, or sequences of text tokens interleaved with other multimodal inputs (state estimates, visuals). Inputs such as images and state estimates are embedded into the same latent embedding as language tokens and processed by the self-attention layers of a transformer-based LLM in the same way as text. 

PaLM-E is also quite data efficient. On the order of 100 examples are needed. This is great news as robotics data is significantly abundant.

#### Key ideas
- Natural language as a representation for planning
	- PaLM-E generates high-level instructions as text, so the model is abel to naturally condition upon its own prediction and directly leverage the world knowledge embedded in its parameters. 
	- Assume that a low-level policy or planner exist to translate the high level decisions into low-level actions. (There are prior works describing how to train such low-level policies, such as Rt-1). The policy can perform low-level *skills* from some small vocab. The plan consists of a sequence of such skills. (action -- skill -- plan)
	- PaLM-E can be understood as a high-level policy that sequences and contriols the low-level policis. 
- Architecture
	- Decoder-only, with multimodal sentence as prefix (prompt). 
	- Encoder and projector to ensure multimodal sentences are encoded into the same space.
	- Inputs: multimodal sentences consisting of text and continuous observations, encoded into the same embedding space.
 	- Output: text 
- Embedding
	- Multimodal info canbe injected into the LLM by **skipping the discrete token level** and directly mapping the continuous observations into language embedding space. 
	- Text token embedding space. From W vocabulary to k-dim space, with projection matrix |W| x k.
	- The continusou observations are encoded into a sequence of vectors (NOT a single vector) with the same dimension as the embedding space of the language tokens. 
- Inputs
	- State estimation: MLP to map into k-dim space
	- Images: a sequence of m k'-dim vectors, then projected to mxk.
	- OSRT (object scene representation transformer), object-centric neural scene representation. Also with a encoding process similar to images. 
	- entity labeling: object instance special token is specified in the input prompts. <obj1>, <obj2>, etc. 


#### Technical details
- PaLM is a base LLM model, similar to GPT-4. Bard is an assistant model based on PaLM. So PaLM <-> GPT-4, Bard <-> ChatGPT. 
- 562B = 540B PaLM + 22B ViT. Language models are typically much larger in terms of parameters which encodes the knowledge of the world.
- Frozen or not?
	- LLMs could be frozen, and encoder has to be trained to produce embedding vectors such that the frozen LLM is grounded on the observations. It is like input-conditioned soft-prompting.
	- However overall finetuning LLM results in better performance. 
- Affordance and failure dectection
	- Affordance: given an image, whether a skill of the low-level policy can be executed in the curr environ.
	- Failure: given an image, was a skill successful?
- 3 types of robot tasks: Task and motion planning (TAMP, in simulation), Language-table dataset (multi-object tabletop pushing environment), and SayCan mobile manipulation (kitchen).

#### Notes
- "Grounding" and "Emboddied". Grounding establish the link between words and percepts. Connecting LLM representations to real-world visual and physical sensor modalities is essential to solving a wider range of grounded real-world problems in CV. 
- Current SOTA visual-language models on VQA cannot handle robotic reasoning tasks. 
- How does PaLM-E compare with [Flamingo](flamingo.md)? PaLM-E directly interleaves visual and language input, where Flamingo augments LLM with visually-conditioned adaptors.
- PaLM-E and [VisionLLM](vision_llm.md) both uses interleaved text and image input. VisionLLM is based on a pretrained LLM (Alpaca), and finetunes with LoRA. It leverages LLM and is able to output low level control signal (bounding box coordidates, etc). The key to this is to **expand the natural language vocabulary**.


#### TODO
* The appendix contains a lot of relevant papers on LLM-based decision making, and robotics. 
* [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)
* [Do Embodied Agents Dream of Pixelated Sheep?](https://arxiv.org/abs/2301.12050) <kbd>ICML 2023</kbd>
