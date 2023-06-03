# [Pix2seq v2: A Unified Sequence Interface for Vision Tasks](https://arxiv.org/abs/2206.07669)

_June 2023_

tl;dr: Extension of Pix2seq to multiple core vision tasks. 

#### Overall impression
The paper showed that a diverse set of core computer vision tasks can also be unified if formulated in term s of a shared pixel-to-sequence interface. Such tasks includes object detection (the only task supported by [pix2seq](pix2seq.md)), instance segmentation, keypoint detection, image captioning.

The formulation of various vision-centric tasks have significant differences in the form of the outputs, customized models with specialized architectures and loss functions are designed for each task. In order to unify them into one **single** model, a **unified** interface has to be created. 

[Pix2seq_v2](pix2seq_v2.md) expands on [pix2seq](pix2seq.md) and defines a new paradigm. Pix2seq is a conditioned sequence **generation** task (conditioned on heterogeneous features other than this token language). Pix2seq_v2 is a conditioned sequence **completion** task.

Many vision-centric tasks can be treated as **image captioning** or **visual question answering (VQA)** in a specific language dialect, a language spoken in the format of specific json schema, for example.


#### Key ideas
- Same architecture with [pix2seq](pix2seq.md), but [pix2seq_v2](pix2seq_v2.md) also conditions on a task prompt so that the model can produce outputs adapted to the task of interest. --> **Essentially it set a start of the sentence and ask the decoder to finish.**
- Unified sequence interface
	- Both task description and outputs are expressed as sequences of discrete tokens.
	- Task prompt start with special tokens such as `[Detect]`, `[Segment]`, `[Keypoint]`, `[Describe]`, and `[Segment]`, `[Keypoint]` tasks are conditioned on a given object instance. The bbox of the object instance is also tokenized and given as part of the prompt, following the practice in [pix2seq](pix2seq.md).
	- For image captioning directly predict text token. **All four tasks share the same vocabulary**, so it combines the vision-centric detection language proposed in pix2seq, and the natural language vocab, which should be in the order of 32k or larger. This is much larger than the vocab size for pix2seq.
- Training
	- Data mixing: multitask batch. Hard to do data aug in a heterogeneous batch.
	- Batch mixing: single-task batch. Adopted by the paper. 

![](https://pic2.zhimg.com/80/v2-f2e971832a536158b666522063a69c51_1440w.webp)

#### Technical details
- Nucleus sampling is used, similar to [pix2seq](pix2seq.md). Alternatives such as beam search can also be used.
- For instance segmentation, multiple (8) samples are independently drawn and averaged (ensembled) to boost performance. 
- For multitask weight tuning is searched greedily by adding one task at a time while keeping the weighting ratio of existing task unchanged. 
- For image captioning, BLEU score is used as KPI.
- Interestingly, the region for keypoint detection is twice the bbox size to provide some background for optimal performance. 
- Other tasks such as OFA and [Flamingo](flamingo.md) focus on higher level tasks where natural language inherently the desired output. This is different from pix2seq focusing on vision-centric task which requires accurate location.


#### Notes
- Questions and notes on how to improve/revise the current work
