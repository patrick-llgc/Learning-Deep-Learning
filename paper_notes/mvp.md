# [MVP: Real-World Robot Learning with Masked Visual Pre-training](https://arxiv.org/abs/2210.03109)

_March 2024_

tl;dr: Large scale visual pre-training helps robotic learning tasks by increasing sample efficiency, achieving better with fewer learning.

#### Overall impression
We can approach representation learning for robotics from two ends: shared representations on the perception side or shared representations on the action side. The focus of MVP is on shared visual representations.

Visual pretraining via a masked autoencoder (MAE), frozen, and then passed into a learnable control module. We train control policies per task, on top of the same frozen encoder for all downstream robotic tasks and embodiments.

Note that this pretraining is vision-only and is NOT multimodal. See follow up work to extend to multimodal pretraining in [RPT](rpt.md), with the frozen vision pretraining from [MVP](mvp.md).

[MVP](mvp.md) generates vision tokens and is essentially one type of continuous vision tokenizer, in contrast with discrete vision tokenizer such as VQ-VAE or MAGVIT-V2.

#### Key ideas
- Pretraining works under 3 conditions
	- Diverse, real-world data.
	- Self-supervision loss.
	- Scalable architecture with transformers.
- Data source
	- Internet and egocentric videos
	- 4.5 million images, sufficiently large and diverse
- Model arch heavily based on MAE.
- MAE pretrianing is better than CLIP pretraining and train tabula rasa.


#### Technical details
- High sample efficiency == low sample complexity. need fewer samples to learn a task.

#### Notes
- [Masked Visual Pre-training for Motor Control](https://arxiv.org/abs/2203.06173), the foundation for this MVP work was not accepted at conf most likely.