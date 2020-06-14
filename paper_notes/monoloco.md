# [MonoLoco: Monocular 3D Pedestrian Localization and Uncertainty Estimation](https://arxiv.org/abs/1906.06059)

_October 2019_

tl;dr: BEV localization for pedestrians with uncertainty.

#### Overall impression
Uses off the shelf human detector and 2D joint detector (Mask RCNN and Pif-Paf). It exploits the relatively fixed height of pedestrians and in particular, shoulder-hip segment (~50 cm) to infer the depth.

The paper also has realistic prediction of uncertainty through aleatoric/epistemic uncertainty. This helps to mitigate those high-risk cases where GT distance is smaller than the predicted one (for which an accident is more likely to happen).

This idea can be readily exploited for mono 3DOD of cars (rigid body with known shape). 

This paper is well written and the quality of the open sourced [code](https://github.com/vita-epfl/monoloco) is amazing! They even have a webcam demo. 

The paper is quite similar to the idea of [DisNet](disnet.md) of using different bbox features to estimate the depth of the object, using a simple MLP. The geometric baseline is 

#### Key ideas
- Intrinsic task error is the localization error due to variation of height. It is estimated through the height distribution in population.
- Uncertainty: 
	- Alleatoric: Laplacian prior inspired aleatoric uncertainty leads to a L1 loss term (instead of L2 term compared to a Gaussian prior)
		- $L = \frac{|1 - \mu/x|}{b} + \log(b) =  e^{-s} |1-\mu/x| + s$
		- Note that the aleatoric uncertainty does not characterize the noise from input image, but rather **the noise from the output from the joint prediction network**.
	- Epistemic: Monte Carlo Dropput
- Algorithm
	- First step to extract 2D joints from image, this helps escaping the image domain and reduce input dimensionality.
	- 2D joint as input to a shallow MLP to predict the distance and associated aleatoric uncertainty.

- **Geometric approach Baseline**: inference using most stable keypoint segment
	- Project each keypoint back to the GT distance to calculate the 3D distance of keypoint segments (head-shoulder, shoulder-hip, hip-ankle). Then the segment with smallest variance is picked to 
	- pif-paf gives better performance than MaskRCNN in geometric baseline. Maybe bottom-up approach gives more accurate estimation.
	- This is exactly what [GS3D](gs3d.md) and [monogrnet V2](monogrnet_russian.md) does. This is one potential direction of improvement for keypoint based approach.

> The main criterion is that the dimension of any object projected into the image plane only depend on the norm of the vector D (x, y, c) and they are not affected by the combination of its components. 


#### Technical details
- The keypoint is projected to normalized image coordinate at z=1 away. This helps to generalize to different cameras. The joints are also zero-meaned to have more generalization.
- Top-down approach (Mask RCNN) and bottom-up approach (Pif-Paf) yields very similar results. 
- Evaluation metrics: ALP (average localization precision), recall at different distance threshold, and ALE (average localization error) in unit of meters. 
- The geometric baseline 
- Stereo method fails for distance after 30 meters. 

#### Notes
- [Monoloco's social distancing branch](https://github.com/vita-epfl/monoloco/tree/social-distance) that can predict pose and orientation of pedestrians. See [project page](https://www.epfl.ch/labs/vita/research/perception/monoloco/)

