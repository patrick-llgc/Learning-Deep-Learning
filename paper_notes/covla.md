# [CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving](https://arxiv.org/abs/2408.10845)

_September 2025_

tl;dr: Adapting a LLM to autonomous driving VLA, by building a VLM dataset.

#### Overall impression
Two goals of using language: 1/ diverse world knowledge and 2/ advanced reasoninig capability, to solve rare and complex scenarios. 

Selling points of the paper
- good documentation of details in the dataset generation pipeline;

Draw backs of this paper
- Sensor suite has front facing camera only.
- Key frame selection logic can be improved. Keyframe is quite delicate topic. See [DriveVLM](drivevlm.md).

The paper reported two interesting behaviors. 1/ the subpar capability of spatial reasoning (left mistaken as right side, etc) and hallucination. 2/ the consistency between langauge and action modality (action error because of CoT error). 

#### Key ideas
- Most rrevious datasets only have high level driving commends (stop, turn left, etc, i.e. meta-actions) but lack detailed trajectory. CoVLA datasets have traj, but not meta actions. 
- Long tail of driving distribution (navigating road closure) necessitates a high level reasoning and decision making capabilityies.
- Two-stage Labeling techniques
	- Rule-based behavior captioning: using classical stack or rule based algorithm to generate the critical objects (traffic lights, lead vehicle) and action (slowing down, etc)
	- VLM-based reasoning caption: Behavior caption and video frames are fed into VLM to generate a reasoning caption. --> Is this really needed?
	- Combination: Behavior caption and reasoning caption are concatenated
- Hallucinatio mitigation: generate free-form caption with structured and factual anchors
	- Extensive rule-based captions as factual constraints 
	- Supplementary multi-turn VQA (one round of question specific about weather, etc) 
- Predicted caption condition is worse than GT caption condition (providing CoT GT)


#### Technical details
- Model: 
	- 7B llama-2
	- CLIP ViT-L as visio encoder (224x224 pixels)
	- ego speed encoder
	- 10x traj token to parallel decoding (following Carllava)
- Data: 80 hours (10k 30s clips), curated from 1000 raw driving logs. 
- Data distribution
	- locations: urban centers, complex highway interchanges, narrow residential streets, mountain winding road, etc
	- Weather: sunny, cloudy, rainy, heavy rain, etc
	- Time of day: daytime, evenining, night, etc
- Data sampling
	- weighting inversely proportiinal to pre-computed empirical distribution (oversampling minority)
	- Steering angle, acceleration, turn signal
- The VLM is Video-llama-2, a large video language model. 
	- Input: 3 seconds video (8 selected out of the 60 frames window)
	- 1 clip 30 sec --> 10 samples annotataed by VLM --> 10x60 combined caption samples. In total 10k x 600 = 6M samples. 
	- This is to assume VLM reasoning caption does not change in the 3 seconds. --> This itself may be problematic too.
	- 8 H100 GPU days to finish

	
	
#### Notes
- A rare paper from a Japanese autonomous driving company [Turing](https://tur.ing/en/technology_e2e). The company specializes in vision only autonomous driving and uses technology BEV and E2E. The Japan version of Wayve. 
- The [HAD dataset](https://arxiv.org/abs/1911.06978) <kbd>CVPR 2019</kbd> paper is really interesting and was manually labeled, and can be viewed as the OG of VLA. Of course it is very different from the current VLA arch but it has the V, L and A components. 
- [Video-llama-2](https://arxiv.org/pdf/2406.07476) is from Alibaba's Damo Academy, and uses STC (spatial temporal convolution) to encode the videos. It also can take audios as input. 
