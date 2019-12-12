# [Probabilistic Object Detection: Definition and Evaluation](https://arxiv.org/abs/1811.10800)

_December 2019_

tl;dr: Proposes a new metric for probabilistic detection.

#### Overall impression
Proposed a benchmark PDQ (probabilistic detection quality) to evaluate probabilistic object detectors.

$$PDQ = \sqrt{DQ * LQ} $$

$$DQ = \exp(-(L_{FG} + L_{BG}))$$
$$L_{FG} = -\frac{1}{|GT|}\sum \log p(TP)$$
$$L_{BG} = -\frac{1}{|GT|}\sum \log (1 -  p(FP))$$
$$LQ = p(class=y)$$

#### Key ideas
- Only pixels in the original mask is counted as TP. Only pixels not in the original bbox is counted as FP. 
- For bbox annotation, we can use the bbox as the mask. 

#### Technical details
- Summary of technical details

#### Notes
- [A Mask-RCNN Baseline for Probabilistic Object Detection](https://arxiv.org/pdf/1908.03621.pdf) provides a benchmark with mask rcnn. The authors change the output of mask rcnn to probabilistic approach by 
	- shrinking bbox by 10%
	- set uncertainty to 20% of width/height

