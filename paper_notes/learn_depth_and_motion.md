# [Unsupervised Monocular Depth Learning in Dynamic Scenes](https://arxiv.org/abs/2010.16404)

_November 2020_

tl;dr: Learning depth and motion without explicit semantic masks.

#### Overall impression
Challenges in SfM: textureless areas, occlusions, reflections, and moving objects. Most previous work deal with dynamic scene by semantic segmentation and ignore pixels in them. [Monodepth2](monodepth2.md) proposed a method to detect pixels do not change across frames to exclude these regions in training.

This paper is the extension of [LearnK](learnk.md) to dynamic scenes. This paper proposed a sparsity regularization on motion field and it seems to work quite well. It learns four parameters, 2 for optical centers (principal points) and 2 for focal length.


#### Key ideas
- Motion network
	- Input (DepthNet output + RGB) * 2 = 8 ch
	- Output 
		- Egomotion M_ego: 6 DoF, 1x1x6.
		- Object motion translational map $T_{obj}(u, v)$, HxWx3. Very sparse, mostly zero.
		- Total motion field $T(u, v) = T_{obj} (u, v) + T_{ego}$
- Motion reg on T_obj(u, v). Note that this is not on the entire motion field $T(u, v)$. 
	- Group smoothness loss: norm of gradients in x and y direction
	- 1/2 sparsity loss: regularization for the residual translation fields. It encourages more sparsity than L1 loss.

- Depth regularization: edge aware smoothness regularization. Regularization is weaker around pixels where color variation is higher.
- Consistency loss: encourages the forward and backward motion between any pair of frames to be opposite of each other
	- $\alpha \frac{|RR_{inv} - \mathbb{1}|^2}{|R - \mathbb{1}|^2 + |R_{inv} - \mathbb{1}|^2} + \beta\frac{|R_{inv}T + T_{inv}|^2}{|T|^2 + |T_{inv}|^2}$
	- Basically we want to penalize (RRinv - 1) and (Rinv T + Tinv), but we need to normalize them.
- RGB loss (photometric loss)
	- Occlusion aware photometric loss
	- SSIM loss

#### Technical details
- Softplus activation for depth: $z(l) = \log(1+e^l)$
- Cityscape contains the most dynamic scenes. KITTI mostly static scenes.
- Motion reg losses encourage a **piecewise constant** object motion field. This makes sense if objects are not rotating or if not fast rotating.
- L-1/2 norm approaches L1 norm for small T motion field value, and approaches 1/2 when larger. This is actually very similar to the "Depth guided L1 loss" in [KM3D-Net](km3d_net.md)

![](https://cdn-images-1.medium.com/max/1600/1*pMQuXXtQSBWo3tZKreeGEQ.png)
![](https://cdn-images-1.medium.com/max/1600/1*RXaL86XRYlaVtbF9MjEeXg.png)

#### Notes
- Questions and notes on how to improve/revise the current work  

