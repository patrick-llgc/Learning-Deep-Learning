# [DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model](https://arxiv.org/abs/2310.01412)

_February 2024_

tl;dr: RT-2 VLA model for autonomous driving.

#### Overall impression
DriveGPT4 offers one solution for end-to-end autonomous driving. It seems to be heavily inspired by [RT-2](rt2.md), from both problem formulation to network architecture.

In a nutshell, it projects multimodal input from image, control into text domain, allowing LLMs to understand and process this multimodal data as text. 

It takes in multiple single-cam images and prompts the LLM to directly output actions. It is in a sense **e2e planning**, without explict modules such as perception stack.

There may be many practical issues with deployment of such a system into production. See Notes session for details.


#### Key ideas
- Architecture: 
	- Overall an VLA model, tailored to AD
	- Vision tokenizer: inspired by Valley, feasuring spatial tokens from each frame, and temporal tokens pooled from multiple frames.
	- Action tokenizer and untokenizer: inspired by RT2.
	- Takes in video frames and human questions, output answers and control signals.
- Dataset
	- BDD-X dataset contains 3 types of questions: action description (what), action justification (why) and control signals (velocity and steering angle). 16K.
	- ChatGPT also used to generate QA pairs for BDD-X dataset, such as traffic lights, and surrounding objects. 40K. --> This part is significantly improved by [DriveVLM](drivevlm.md), with a more diverse and structured CoT prompting.
	- General vision-language dataset: 600K video and 150K image.
- Training
	- Pretraining: train projection module only with general vision-language dataset, for vision-language alignment. Similar to the scheme of [Llava](llava.md). --> Why not use one VLM off-the-shelf?
	- Mix-finetuning: finetune the entire system. This comes from the "Co-Fine-Tuning" technique by RT2. Interestingly, this improves not only the QA performance, but also improves action prediction performance. In summary, more diverse data is better.
- Eval: with chatGPT to give similarity scores. --> There is a more structured and improved version in [DriveVLM](drivevlm.md).


#### Technical details
- Predicts immediate action for one step (current time stamp). Understandably, this is inspired by [RT-2](rt2.md). In contrast, [DriveVLM](drivevlm.md) predicts future planning waypoints, and use a faster replanner to refine and solve for control. --> **This is one key advantage of planning in the form of waypoints instead of action**. 

#### Notes
- The paper seems to be written in a hurry, with many things to improve:
	- Latency not reported. Potentially very large, on the order of seconds.
	- Only uses front camera, yet this is not sufficient to ensure safe driving. For example, it is unable to handle rapid overpassing objects from behind during a lane change. 
	- The selection of threshold in section 5.2 seems arbitrary for speed and turning angle, i.e., why would 1 m/s and 1 deg assume the same importance, given their diff physical unit?
	- The paper lacks systematic comparison with other SOTA E2E methods, such as UniAD.
	- Misc typos still not corrected at v3: Page 2 (wrong citations for first DL work on E2E AD, DriveLM should cite the paper, etc)
- Explanable AD, dataset with video-language pairs.
	- Talk2Car 2019
	- ADAPT 2023
	- DRAMA 2023
