# [Feature-metric Loss for Self-supervised Learning of Depth and Egomotion](https://arxiv.org/abs/2007.10603)

_August 2020_

tl;dr: Feature metric loss to avoid local maxima in monodepth.

#### Overall impression
Local minima in monocular depth estimation happens as it is sufficient but not necessary for small photometric error. This issue has been tackled by either replacing photometric with feature-metric errors, or using cues to guide optimization out of local minima ([Depth Hints](depth_hints.md) and [MonoResMatch](monoresmatch.md)). 

In comparison, [Depth Hints](depth_hints.md) still uses photometric loss, and [Feature metric monodepth](feature_metric.md) will largely avoid the inferenece of local minima.

The discussion of feature metric loss is perhaps first raised in [BA-Net](banet.md) and [Deep Feature Reconstruction](depth_vo_feat.md). It has the advantage to be less sensitive to photometric calibration (camera exposure, white balance) and is dense supervision. 

However how to learn this feature map is the key. The paper uses AutoEncoder to do this, and have two extra loss terms to ensure large but smooth gradient, for faster and more general optimization. 

>> Small photometric loss does not necessarily guarantee accurate depth and pose, especially for pixels in textureless region. Depth smoothness loss forces depth propagation from discriminative regions to textureless regions. However such propagation is with limited range and tend to cause over smooth results. 

>> A set of assumptions (for SfM-Learner): the corresponding 3D point is static with Lambertian reflectance and not occluded in both views. 

#### Key ideas
- Learn a good feature
	- Use AutoEncoder to learn the encoded feature. 
	- **Discriminative loss** encourages gradient in texture region.
	- **Convergent loss** encourages the gradient to be smooth, and thus ensures a large convergence basin. 
	- In summary, the feature has large first order but small second order gradients. The discriminative loss and convergent loss combined lead to a smooth sloped feature map in textureless region.
- The feature-metric loss is combined with photometric loss. Not sure how this changes when feature-metric loss is used alone.
- Online refinement for 20 iterations on one test sample.

#### Technical details
- Both $\partial L/\partial D(p)$ (depth) and $\partial L/\partial G$ (pose) rely on image gradient $\partial I/\partial p$. For texture-less regions, the image gradients are close to zero and thus contributes to zero loss for depth and pose. Thus we need to learn a better feature representation $\phi$ to solve this issue such that $\partial \phi/\partial p$ is not zero.
- [DORN](dorn.md) and [BTS]() are still the SOTA for supervised monodepth.
- Depth normalized by depth mean in loss function.

#### Notes
- In retrospect, performing photometric loss is quite fragile and dangerous. Photometric calibration (required by DSO and SfM-Learner) is perhaps as simple as one layer of neural network and we should leave this to the network to learn a good feature to use for depth estimation.

