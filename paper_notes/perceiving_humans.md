# [Perceiving Humans: from Monocular 3D Localization to Social Distancing](https://arxiv.org/abs/2009.00984)

_September 2021_

tl;dr: Improved version of Monoloco (monoloco++) and application in social distancing.

#### Overall impression
This paper builds upon the previous work of [MonoLoco](monoloco.md).

The low-dimensional representation of humans give it more generalization. It escapes the image domain and reduce the input dimensionality. This makes the skeleton-baesd network extremely fast to train (2 min on a single 1080Ti GPU card).

Monoloco++ beats mono3D baselines (such as [SMOKE](smoke.md)). They have roughly the same performance on easy cases, but much better performance in medium/hard cases. And monoloco++ has higher recall than SMOKE (39% --> 70%).

#### Key ideas
- Difference between monoloco++ and [MonoLoco](monoloco.md)
	- Multi-task prediction of combining 3D localization, orientation and bbox. [monoloco](monoloco.md) only predicts the 3D localization (distance and its uncertainty).
	- Use of spherical coordinate system (why use radial distance instead of depth?)
- Social interaction and distancing. People tend to arrange themselves spontaneously in a specific configuration called **F-formation**.
	- o-space (free space between a group of people)
	- p-space (donut space enclosed by concentric rings covering people)
	- r-space (beyond the concentric rings)
	- The presence of F-formation is based on finding that two people are within a certain distance, their o-space is not intruded, and looking inward the o-space.

#### Technical details
- The ill-posed problem of 3D localization of human, the task error roughly scales linearly, is about 1 meter in 20 meters.
- Laplacian loss (with aleatoric uncertainty) works roughly the same as L1 loss.

#### Notes
- Questions and notes on how to improve/revise the current work  

