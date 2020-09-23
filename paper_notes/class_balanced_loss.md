# [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555)

_September 2020_

tl;dr: Calculate effective numbers for each class for better weighted loss.

#### Overall impression
This paper reminds me of [effective receptive field](https://arxiv.org/abs/1701.04128) paper from Uber ATG, which basically says the effective RF grows with sqrt(N) with deeper nets. 

This paper has some basic assumptions and derived a general equation to come up with the effective number for weight. The effective number of samples
is defined as the volume of samples and can be calculated
by a simple formula $(1−\beta^N)/(1-\beta)$, where N is the number
of samples and $\beta \in [0, 1)$ is a hyperparameter.

People seem to have noticed it and uses some simple heuristics to counter the effect. For example, this paper noticed using 1/N would bias the loss toward minority class and thus simply uses 1/sqrt(N) as the weighting factor, in [PyrOccNet](pyroccnet.md).

#### Key ideas
- Summaries of the key ideas

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

