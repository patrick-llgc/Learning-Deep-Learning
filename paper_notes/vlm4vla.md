# [VLM4VLA: Revisiting Vision-Language-Models in Vision-Language-Action Models](https://arxiv.org/abs/2601.03309)

_February 2026_

tl;dr: VLM is important initializaing for VLA but vision encoder needs finetuning.

#### Overall impression
Most Surprising Finding is that the performance requirements for VLMs in embodied manipulation tasks do not fully align with their VQA capabilities. Specifically, and contrary to common expectations, VLMs that perform well on general VQA benchmarks are not necessarily better when used in VLAs. Furthermore, on various auxiliary Embodied-QA tasks, we discover that fine-tuning on most of these tasks leads to a performance degradation in the resulting VLA.

QwenVL series models significantly outperform
other VLMs.

#### Key ideas
- To our suprise, our findings reveal that while VLM initialization offers a consistent benefit over
training from scratch, a VLM’s general capabilities are poor predictors of its downstream task
performance, contrary to common assumptions.
- We find that fine-tuning the vision encoder is essential for strong control performance,
whereas the language encoder is less critical.
- the VLA is trained from scratch without any
pretrained VLM, indicating that VLM pre-training is fundamental to the ability of VLA models.
- A significant gap exists between the capabilities required for VLA manipulation tasks and those measured by existing VQA benchmarks. 
- finetuning vision encoder
	- finetuning the vision encoder is crucial when adapting a VLM into a VLA, and that the impact of this module can be more significant than merely increasing the number of trainable parameters in the language model. 
	- We observe a significant performance degradation for all models on both the Calvin and Simpler benchmarks after freezing the vision encoder
	- When it is frozen, the total number of tunable
parameters is still a substantial 7.6B, which is much larger than 3.8B in Qwen2.5VL-3B. However, the performance of the frozen Qwen2.5VL-7B is not only significantly worse than its fully finetuned
counterpart but also substantially underperforms the fully finetuned Qwen2.5VL-3B.
	- whether the word embeddings are trained or kept fixed has no noticeable impact on VLA performance.
	- This shows a substantial mismatch between the intrinsic visual representations learned by VLMs pretrained on large-scale web image–text corpora and the visual signals required in manipulation settings.


#### Technical details
- A simple MSE loss is used. The flow matching or diffusion losses introduce significant stochasticity during inference, requiring a much larger number of rollouts for accurate evaluation.
- It is also worth noting that finetuning with generation tasks (i.e., Omni-Generation on
Qwen2.5VL-7B), such as depth and semantic map prediction, did not yield performance ben-
efits. This may indicate that simply introducing generation tasks or dense 3D-aware tasks into VLM
finetuning process does not provide a tangible advantage for the VLA.

#### Notes
- <!--Questions and notes on how to improve the current work-->

