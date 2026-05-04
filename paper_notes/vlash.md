# [VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference](https://arxiv.org/abs/2512.01031)

_May 2026_

tl;dr: Async inference with improved smoothness, by taking in observation at t0 and estimated ego future stage at t0 + delta_t.

#### Overall impression
Another great innovation from Zhijian's group! This is the SOTA technique for async inference so far.

#### Key ideas
- Future ego state awarenes
	- Train and inference with o(t) and s(t+δ), given a delay δ, and predict a(t+δ:t+δ+H)
	- Train with a random δ during training
	- Efficient training with shared o(t) and multiple (s, a) pairs to force the network to pay attention to the random δ. (VLA in robotics has a tendency to overly pay attention to the image.) 
	- The paper also comes up with a smart mask to do multiple training in one forward pass, to boost training efficiency.
- Action Quantization: essentially binning nearest q action steps into 1. This is the main source of acceleration in this paper.

#### Technical details
- <!--Summary of technical details, such as important training details, or bugs of previous benchmarks.-->

#### Notes
- **Potential issue**: If s^{GT}_{t+δ} contains information caused by future observations or interventions that are unavailable at t, the model may learn a non-causal dependence on future state. In othere words, there would be an **info leakage**, or train/inference mismatch. It may teach the planner to rely on a future ego state that implicitly knows about future scene changes.
	- The mismatch relies on training with `s_GT(t+δ)` but infer on `s_rollout(t+δ)`.
	- For robotics where the environments are static or semi-static, this leakage may be OK. 
	- For Autonomous Driving (AV) events, issues are more problematic. 
- A potential fix
	- train with mixture of 
		- A. GT future ego: input = o(t), s_GT(t+δ)
		- B. rollout future ego: input = o(t), s_rollout(t+δ)
		- C. corrupted/noisy future ego: input = o(t), s_rollout(t+δ) + noise
	- Validate with scenarios with surprise events during [t, t+δ]
- Another fix is to filter surprise dataset altogether as one may argue these data leads to non-causal learning and are "poisonous" data for this formulation
- And of course, this may not be a huge issue afterall, but hard to tell without doing actual experiment. 