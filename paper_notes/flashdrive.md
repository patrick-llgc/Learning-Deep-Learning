# [FlashDrive: Flash Vision-Language-Action Inference For Autonomous Driving](https://openreview.net/pdf?id=kuZrNI5oZM)

_April 2026_

tl;dr: Accelerating VLA model in streaming fashion.

#### Overall impression
The paper broke down the inference pipeline into four parts: vision ecnoder, prefill, decode and trajectory decode.

#### Key ideas
- Q: “If the old frame is the same old frame, why can’t I just reuse its K/V exactly?”
- A: In a sliding window session, although the input frame stay the same, the tranformer context has changed as the older frames have been dropped. 
	- Example: Old window: [A, B, C, D], Next window: [B, C, D, E]
	- Cached D still carries some influence from A, fresh D would not. So reused K/V is stale.
	- This would lead to a train/inference mismatch.
- streaming fine-tuning means:
	- “Train the action head on these approximate caches until it becomes robust to them.”

#### Technical details
- <!--Summary of technical details, such as important training details, or bugs of previous benchmarks.-->

#### Notes
- <!--Questions and notes on how to improve the current work-->

