# [monoDR: Monocular Differentiable Rendering for Self-Supervised 3D Object Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660511.pdf)

_September 2020_

tl;dr: Use differentiable rendering for monocular 3D object detection, without any 3D labels. 

#### Overall impression
The gist of the paper is how to perform 3D object detection without 3D annotation. The answer is to use differentiable rendering to form self-supervised constraints with 2D annotations. 

The scale ambiguity due to projective entanglement of depth and scale are handled by explicitly predicting the metric size of objects. [monoDR](monodr.md) uses a size loss penalizing the reconstructed loss from deviating too much of the averaged size of the object class. Concurrent work of [SDFLabel](sdflabel.md) uses lidar to recover the absolute scale.

It uses an analysis-by-synthesis methodology which is similar to [3D RCNN](3d_rcnn.md) and [RoI10D](roi10d.md), and [SDFLabel](sdflabel.md).

The paper also provides an interesting descent-and-explore approach to avoid local minima, most likely by using the so-called **hindsight loss**.

#### Key ideas
- Architecture predicts 
	- location (XYZ), dimension (HWL), rotation (YPR)
	- shape (latent vector h_s)
	- texture (latent vector h_t)
- Differentiable Rendering via [Neural 3D mesh renderer](https://github.com/hiroharu-kato/neural_renderer).
- Loss
	- Mask (silhouette) loss
	- bbox loss, allowing a margin up to t (loss with flat bottom)
	- distance loss (compare with [packNet](packnet.md))
	- projection loss (photometric loss)
- Mask loss alone will be not very sensitive to rotation, leading to large estimation error to rotation.
- Shape encoder and decoder are trained from ShapeNet dataset, down to a **8-dim** vector.
- 3D conf is 2D conf modulated by self-consistency. This improves performance. 

#### Technical details
- Latent space of car shapes 
	- [3D RCNN](3d_rcnn.md) 10-dim
	- [RoI10D](roi10d.md) 6-dim
	- [monoDR](monodr.md) 8-dim
- Only 2D annotation is not good enough. Even noisy supervion from depth is essential for 3D monocular object detection.
- **Escaping local minima**: Estimating rotation with render and compare approaches can easily lead to local minima. --> this is implemented as a "hindsight loss", initially developed for multi-choice learning. It leverages the fact that the function min is a differentiable way to ensemble multiple differentiable choices.
	- [Unsupervised Learning of Shape and Pose with Differentiable Point Cloud](https://arxiv.org/abs/1810.09381) <kbd>NIPS 2018</kbd>
	- [Multi-view Consistency as Supervisory Signal for Learning Shape and Pose Prediction](https://arxiv.org/abs/1801.03910) <kbd>CVPR 2018</kbd>
- The bbox used in the work seems to be modal. The estimation of 3D objects need to be amodal to get worked properly.
- The paper argues that better segmentation results does not necessarily leads to better results. --> but why?

#### Notes
- Questions and notes on how to improve/revise the current work  

