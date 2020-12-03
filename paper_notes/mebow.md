# [MEBOW: Monocular Estimation of Body Orientation In the Wild](https://arxiv.org/abs/2011.13688)

_June 2020_

tl;dr: Monocular estimation of body orientation.

#### Overall impression
The paper annotates body orientation in COCO dataset. 

[TUM dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/people-detection-pose-estimation-and-tracking/monocular-3d-pose-estimation-and-tracking-by-detection/) has 8 bins, and got later extended to continuous labels by averaging annotations from 5 labelers. 

#### Key ideas
- 72 bins x 5 deg interval for annotation. This is still within the human cognition limit.
- The circular gaussian loss is used to **blur the one-hot classes and then regresses a heatmap**. Interestingly the paper uses L2 loss directly instead of a BCE loss.
	-  $p \propto e^{-\frac{1}{2\sigma^2}(\min(|i-gt|, 72-|i-gt|))}$
	- The loss is approximation [von Mises distribution](https://en.wikipedia.org/wiki/Von_Mises_distribution) (see [mono3D++](mono3d++.md) which actually used this idea to regress angles in 360 bins).
- The paper also formalized the definition of body orientation by TxS, perpendicular to both torso direction and shoulder direction. This is needed to incorporate body orientation as a weak supervision to 3d body estimation.

#### Technical details
- Labeling tool design. It is a brilliant idea to show case example images with body orientation. (This could be done for car orientation estimation as well).
![](https://cdn-images-1.medium.com/max/1440/1*f0y93g3RBeWUqKNWyGcLIg.png)

#### Notes
- The version with appendix can be found on [amazon](https://assets.amazon.science/30/28/2bdee5464430ae374176fd77d326/scipub-1311.pdf).
- Classification over regression: Sometimes it is useful to convert regression to a multi-class classification with ordered bins. Instead of directly predicting a one-hot label, it blurs the one hot label by allowing leakage into its neighboring bins. This is similar to the idea of label smoothing to stabilize training. I feel that this is actually a particularly useful technique for multi-bin classification, when the bin numbers are very large, and when bins are ordered.
- [Designing Deep Convolutional Neural Networks for Continuous Object Orientation Estimation](https://arxiv.org/abs/1702.01499) proposed three different methods to regress a continuous angle.