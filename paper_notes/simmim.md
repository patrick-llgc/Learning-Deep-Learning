# [SimMIM: A Simple Framework for Masked Image Modeling](https://arxiv.org/abs/2111.09886)

_November 2021_

tl;dr: Large scale pretraining based on Masked Image Modeling. Similar to MAE.

#### Overall impression
This paper is published a week after [MAE](mae.md), obviously rushed by the publication of the latter. The ideas are very similar, but execution (hyperparameter tuning, paper writing) is considerably inferior to MAE.


Difference between [MAE](mae.md) and [SimMIM](simmim.md):

- MAE uses asymmetric design of encoder and decoder, where encoder does not see masked patches. SimMIM uses symmetric design.
- SimMIM stressed the difference between prediction (of only masked patches) and reconstruction (of all patches), and mentioned that the former yields better performance. MAE also observes the trend (in footnote). However MAE also demonstrates the mid-ground: training without losses on visible patches but prediction on all the patches.
- SimMIM was not validated on more fine-grained downstream tasks such as object detection and segmentation.

Similarities between [MAE](mae.md) and [SimMIM](simmim.md):

- directly regress the pixels
- light decoder design

#### Key ideas
- Summaries of the key ideas

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work
