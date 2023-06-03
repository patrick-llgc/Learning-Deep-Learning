# [VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks](https://arxiv.org/abs/2305.11175)

_May 2023_

tl;dr: Use LLM as flexible and unified manager for a variety of vision tasks. 

#### Overall impression
All vision foundation models are restricted to tasks in a pre-defined form, struggling to match the open-ended task capability. VisionLLM aims to flexibly manage vision-centric tasks (obj det, instance seg, etc) with language instructions.

The main contribution is in the decoder, and seems to be a more user-friendly way to manage/unify multiple tasks than [pix2seq v2](pix2seq_v2.md), and can generate to multi-modal input.

The "output format as query" trick seems a nice way to speed up inference, but it breaks the beauty of the next-token prediction paradigm and has to resort to inductive bias or prior knowledge of specific tasks.


#### Key ideas
- Three parts
	- A unified language instructions for vision-language tasks (a **flexible and uniform** task description format)
	- Language-guided image tokenizer. 
	- LLM-based task decoder. 
- Architecture details
	- The visual perception tasks did share one common format (C, P).
	- Language-guided image tokenizer.  generates language-aware visual features --> How important is this language-awareness? It is important for visual grounding but not for object detection.
- Output-format-as-query
	- this trick in LLM-based task decoder helps solving the slow inference problem of autogressive (or causal, as the paper calls it) models. The output format acts as slots to be filled in, and these queries can generate output all at the same time.
	- However, this formulation breaks the beauty and simplicity of using next-token supervision, and has to resort to bipartite matching as in [DETR](detr.md).
- Open-endedness and generalization
	- Task description is human-like. Trained with self-instruct to generalize to multiple related or equivalent format. 
	- The model can generalize to different numbers of segmentation points (with 8 points vs 16 points), and different classes in object detection (detecting more classes, or ignoring more classes), according to a category set (dict).

#### Technical details
- Other related tasks
	- A different way to manage expert models is via APIs like [HuggingGPT](hugging_gpt.md). It is not quite end-to-end as VisionLLM.
	- Visual prompt tuning: gives image and masks as demonstrations. The generalization of the task format to various real-world problem is hard. We need a more generalized format. 
	- Previous works focus on image to text tasks (such as [Flamingo](flamingo.md), OFA) but not on vision-perception tasks. 
- Visual grounding: locate objects in an image according to natural language discription.
- VisionLLM achieves performance on object detection on-par with detection-specific models.


#### Notes
- The LLM-based image tokenizer and LLM-based task decoder share the same description of the task.
	- The effectiveness and necessity of the language-guided image tokenizer is unclear. --> Can we just use the normal image feature as condition, like in [pix2seq](pix2seq)? The LLM-based task decoder is indeed very useful and is a user-friendly way to manage vision-centric tasks with enough flexibility.
	- How does [pix2seq v2](pix2seq_v2.md) unifies multiple tasks?
