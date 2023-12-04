# [RoboVQA: Multimodal Long-Horizon Reasoning for Robotics](https://arxiv.org/abs/2311.00899)

_December 2023_

tl;dr: RoboVQA proposes a scalable, bottom-up and intrinsically diverse data collection scheme. Egocentric human data helps. 

#### Overall impression
RoboVQA demonstrates a substantial gap in performance for zero-shot SOTA LLM/VLM models compared to the model finetuned on real embodied data. This indicates a critical need to collect tons of grounded data. RoboVQA proposes a **scalable** bottom-up data collection scheme.

RoboVQA proposes a bottom-up long-horizon data collection, as compared to the tradiational top-down step-by-step data collection. It also proposes to use humans to collect egocentric videos. By combining the two, RoboVQA can achieve 14x speedup. The collected data are also highly diverse, relevant and on-distribution for users. (采集数据又快又好)

#### Key ideas
- Data collection procedure
	- Given long-horizon user requests (such as make coffee)
	- human operator teleoperats a ronbot to fullfil task
	- Medium-horizon tasks are labeled in hindsight (such as put coffee on table) via crowd-sourcing, with temporal segmentation and task instructions for each segment. 
	- For each sequence of medium-horizon labeled segments, 10 types of QA pairs are generated. 
- Top-down vs bottom-up
	- More traditional collection uses a fixed and small list of tasks decided in advance by researchers and engineers in a top-down fashion
	- Bottom-up approach where many tasks are crowded-sourced by users and teleoperators.
- Eval of intervention rates (lower the better)
	- Cognitive intervention, high level text domain
	- Physical intervention, low level motor command domain
- Core model
	- RoboVQA-VideoCoca, finetuned from VideoCoca, a video VQA, or video captioning model.
	- 0.3B, still better than the 30x larger (12B) PaLM-E model.
- Three embodiments, robots, human, human using a grasping tool.
- Dataset
	- 238 hours of video (10 days), 5000 long-horizon, 90,000 medium horizon, and **800,000 (nearly a million) video-text QA pairs**. 
	- Long horizon averages at 102 seconds
	- Medium horizon averages at 14 seconds
	- Eval set small, 1000 VQA entries, as evaluation is by humans.

#### Technical details
- Task augmentation matters. Ask 10 questions per segments, and train a multi-task head.
- Cross-embodiment transfer --> How useful is human data?
	- Human data helps! By itself is useful to acquire grounded understanding of videos with robot embodiment.
	- Human data does not hurt! Extra data with human embodiment does not hurt performance when evaluating on robot embodiment. 

#### Notes
- Very similar to the Ego4D dataset. Ego4D has egocentric videos of daily human activigties annotated with dense narrations.
- The appendix gives a good collection of long-horizon and medium-horizon instructions.