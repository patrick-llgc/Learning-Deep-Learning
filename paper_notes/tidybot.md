# [TidyBot: Personalized Robot Assistance with Large Language Models](https://arxiv.org/abs/2305.05658)

_July 2023_

tl;dr: Use LLM to summarize and apply user preference through few shot learning. This provides a great way to personalize/customize LLM.

#### Overall impression
TidyBot achieved personalization of LLM through few-shot summarization. The context learning from few examples enabled some user customization, without fine-tuning of LLM. It is essentially a nice piece of **prompt engineering**, and promotes LLM's [Chain-of-Thought](cot.md).  It is reasonable to enable customization of LLM through prompt engineering rather than fine-tuning the base model.

The combination of **summarization** and **open-vocabulary classification** is critical to the autonomy of TidyBot. This enables object classifier to work with a small set of generalized object categories. **Summarization provides a good way of generalization.**

Yet another key question is how to form data close loop to continuously improve the LLM? This must be done through parameter efficient tuning of LLM (LoRa, etc). This is also a bit like the supervised finetuning (SFT) of ChatGPT. 

The design of a benchmark dataset is quite informative and provides best practice for robotics research.

Overall this makes a great high-school project. 

#### Key ideas
- Use LLM off-the-shelf with no additional training of data collection. --> Thus prompt engineering to do customization
- One-size-fit-all plan would not address the wide variability in user preference. 
- User examples are converted to Pythonic code as LLM prompts. This provides a structured output that is easy to parse, following Socratic Models and ProgPrompt. 
- Offline learning of personalized receptacle and primitive
	- Give few shot examples and learn summary (preferred receptacles and primitives)
- Online System design
	- Localize nearest object via oracle perception
	- Classify object into generalized category
	- Determine appropriate receptacle and manipulation primitive using generalized rules produced by LLM.
	- Use manipulation primitive to put object into receptacle.
- LLM summarization automatically provide candidate categories. This is used as target label set for CLIP.
	- Cosine similarity between summarized object category and image features in CLIP embedding space. 

#### Technical details
- TidyBot is a real-world mobile manipulator. 
- Receptacle (proper place to hold things)
- Open vocabulary image classifier vs open vocab object detector
	- cls: CLIP
	- obj det: ViLD, OWL-ViT
	- open vocab obj det sometimes fail to detect foreground object even through it is prominently in the center of the image. 
	- Open vocab cls always gives a prediction, but sometimes gives the class of the background objects. --> This can be improved with SAM.
- Base model LLM: GPT-3. Text-Davinci-003
- Error breakdown
	- perception: Object localization
	- Object classification
	- Execution 96% success rate, with 15-20s per object.
- HW
	- Power-Caster Drive System 3-DoF
	- Kinova Gen3 7-DoF arm 
	- Robotiq 2F-85 parallel jaw gripper
- Simplified perception and localization
	- Two overhead camera for robot pose estimation and object localization
	- ArUco fiducial marker
	- Planning is to find shortest collision-free path and occupancy map. Pure-pursuit algorithm to follow computed path.
- Ablation study shows that the results using text embedding to find closest known category is also quite good. 
- LLM exhibits remarkable commonsense reasoning abilities. Yet sensible choice does not necessarily reflect personal preference. 

#### Notes
- Questions and notes on how to improve/revise the current work
