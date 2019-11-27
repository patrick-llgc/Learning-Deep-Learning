# [Mono3D++: Monocular 3D Vehicle Detection with Two-Scale 3D Hypotheses and Task Priors](https://arxiv.org/abs/1901.03446)

_August 2019_

tl;dr: Mono 3DOD based on 3D and 2D consistency, in particular landmark and shape recon.

#### Overall impression
The paper is written in overcomplicated math formulation. Overall not very impressive. The consistency part is quite similar to other papers such as [deep3dbox](deep3dbox.md).

The morphable wire frame model is fragile and the authors did not do a thorough ablation study on its contribution. I am not sure if shape recon is a good idea, especially to handle corner cases. --> **Nobody in the literature actually talks about how to handle corner cases. This need to be acquired through engineering practice.** Maybe CV method is needed to handle the corner cases. 

The paper seems to use 3D depth off the shelf but it was not described in details.

#### Key ideas
- Learn a morphable wire model from landmarks (takes 2.5 min, deterministic). --> similar to [ROI 10D](roi10d.md).
- Metrics: ALP (average localization precision). This metric only cares about center location.

#### Technical details
- w and h are encoded to be exponential forms because they need to be positive. 

#### Notes
- Where does the label come from?
- The wireframe model is fragile and cannot model under-represented cases.
- von Mises distribution: circular Gaussian distribution