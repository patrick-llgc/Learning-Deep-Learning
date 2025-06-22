# [Tokenize the World into Object-level Knowledge to Address Long-tail Events in Autonomous Driving](https://arxiv.org/abs/2407.00959)

_June 2025_

tl;dr: uses [para-drive](para-drive.md) as agent-centric tokenizer to generate **object level latent tokens** with LLM to enhance AD.

#### Overall impression
Feeding sparse tokens from BEV perception to LLM seems to be an effective way to leverage LLM.

The model leverageds LLMs's reasoning capability to enhance AV planning in long-tail scenarios.

LLM based motion planning, as prev generation of LLM-based motion planner, formulate motion planning as a language modeling problem (like [GPT-driver](gpt_driver.md)). It converts ego stats and observations into language prompts. It depends on the **quality and resolution of the scene description**. Designing templates to textualize scenes requires extensive prompt engineering.

[Driving with LLMs](driving_with_llms.md) from Wayve uses explicit symbolic representations to enecode scene information, which are then used as tokens for LLMs. In comparison, [TOKEN](token_ad.md) uses implicit object-level tokens. 

The main challenge [TOKEN](token_ad.md) solves is how to train an effecitive scene tokenizer in the low-data regime. The key answer is to leverage perception or object level representation from existing end to end driving solutions (such as PARA-Drive).

#### Key ideas
- Para-Drive as scene tokenizer to extract object-level tokens. 
- 3 stage training
	- **representaition alignment**: only train adaptor, with perception QA 
	- reasoning alignment: train adaptor and Lora, with reasoning and planning QA
	- planning alignment: train adaptor and Lora, with planning QA
- 3 kinds of QA datasets
	- perception/understanding QA dataset
	- behavior reasoniong QA (critial object reasoning)
	- planning dataset
- Long-tail eval scenarios
	- construction sites
	- executing 3-point turns
	- resume motion after full-stop
	- overtaking parking cars through oncoming lane (out of lane nudging)
- General eval and long-tail eval. Reasoning only helps with long-tail eval scenarios. 
- Evoking common-sense reasoning in a LLM-powered planner requries proper alignment rather than just a larger LLM backbone.


#### Technical details
- input/output
	- 4 frames x 0.5s = 2s of history
	- 6 x 0.5s = 3s of motion plan

	
	
#### Notes
- The paper was not clear if there is an image/video tokenizer used together with Para-drive tokenizer. If not, then this method is not an end-to-end method. The paper mentioned that "in addition to the object-level tokens, unstrauctured scene-level latent tokens learned from scratch can be *optionally* included to compensate for missing information, such as weather conditions".

