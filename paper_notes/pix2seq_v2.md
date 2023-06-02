# [Pix2seq v2: A Unified Sequence Interface for Vision Tasks](https://arxiv.org/abs/2206.07669)

_June 2023_

tl;dr: Extension of Pix2seq to multiple core vision tasks. 

#### Overall impression
The paper showed that a diverse set of core computer vision tasks can also be unified if formulated in term s of a shared pixel-to-sequence interface. Such tasks includes object detection (the only task supported by [pix2seq](pix2seq.md)), instance segmentation, keypoint detection, image captioning.

The formulation of various vision-centric tasks have significant differences in the form of the outputs, customized models with specialized architectures and loss functions are designed for each task. In order to unify them into one **single** model, a **unified** interface has to be created. 

Many vision-centric tasks can be treated as **image captioning** or **visual question answering (VQA)** in a specific language dialect. (A language spoken in the format of specific json schema.)


#### Key ideas
- Same architecture with [Pix2seq](pix2seq.md), but pix2seq_v2 also conditions on a task prompt so that the model can produce outputs adapted to the task of interest. --> **Essentially it set a start of the sentence and ask the decoder to finish.**
- Other tasks such as OFA and Flamingo focus on higher level tasks where natural language inherently the desired output. This is 

#### Technical details
- Nucleus sampling is used, similar to [Pix2seq](pix2seq.md). Alternatives such as beam search can also be used.
- For instance segmentation, multiple (8) samples are independently drawn and averaged (ensembled) to boost performance. 
- For multitask weight tuning is searched greedily by adding one task at a time while keeping the weighting ratio of existing task unchanged. 
- For image captioning, BLEU score is used as KPI.
- Interestingly, the region for keypoint detection is twice the bbox size to provide some background for optimal performance. 

#### Notes
- Questions and notes on how to improve/revise the current work
