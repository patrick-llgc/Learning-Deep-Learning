# [Depth Hints: Self-Supervised Monocular Depth Hints](https://arxiv.org/abs/1909.09051)

_July 2020_

tl;dr: Use depth pseudo-label to guide the self-supervised depth prediction out of local minima.

#### Overall impression
This paper digs into self-supervised learning and provides tons of insights, in a fashion similar to [What Monodepth See](what_monodepth_see.md).

It first showed that the photometric loss function (DSSIM + L1) used in monodepth can struggle to find global minimum and get trapped in local minima during training. Then it provides a way to effectively use depth pseudo-label, in a soft supervised way. Depth hints are used when needed to guided the network out of local maxima. --> In a way, it is similar to the idea of using the minima of reprojection loss from multiple frames as in [Monodepth2](monodepth2.md).

This paper proposed a way to consume possibly noisy depth label together with self-supervised pipeline, and is better than using supervised signal alone, or simply sum the two loss together.

Another way to avoid local maxima is to use feature-metric loss instead of photometric loss, such as in [Feature metric monodepth](feature_metric.md), [BA-Net](banet.md) and [Deep Feature Reconstruction](depth_vo_feat.md). In comparison, [Depth Hints](depth_hints.md) still uses photometric loss, and [Feature metric monodepth](feature_metric.md) will largely avoid the inferenece of local minima.

Both [Depth Hints](depth_hints.md) and [MonoResMatch](monoresmatch.md) propose to use cheap stereo GT to build up monodepth dataset. [Depth Hints](depth_hints.md) uses multiple param setup to obtain an averaged proxy label and use a soft (hint) supervision scheme. [MonoResMatch](monoresmatch.md) uses left-right consistency check to filter out spurious predictions and a traditional hard supervision scheme. 

#### Key ideas
- When we have pseudo-label (proxy label), we can use it in the following way
	- $l_r$ is photometric reprojection loss, $l_s$ is supervised loss
	- self-supervision: $l_r(d)$
	- hard supervision: $l_s^{\log L1} (d, h)$
	- sum supervision: $l_{sum} = l_r(d) + l_s^{\log L1} (d, h)$
	- soft (hint) supervision: $l_{hint} = l_r(d) + l_s^{\log L1} (d, h)$, if $l_r(h) < l_r(d)$, else $l_r(d)$
	- hard supervision with uncetainty
- Using SGM pseudo-label with conventional supervision scheme actually hurt performance.
- For noisy depth, uncertainty prediction is also enabled. This is also used in [SfM from SfM](learn_sfm_from_sfm.md). However using uncertainty with hard supervision is still not as good as depth hints in depth prediction.

#### Technical details
- SGM (semi-global matching) as pseudo-label (available in openCV). Randomly sample hyperparameters as guidance.
- The only way to deal with local minima seems to be use better formulated loss, or use depth hints. 
- Postprocessing of depth map: TTA of horizontally flipped image, then average
- Depth loss is usually done on 1/d or disparity
- log L1 loss: $\log(1 + |d_i - h_i|)$. It works well with estimated depth (may be noisy)
- berHu loss: L1 loss when small, L2 loss when big. Usually used for lidar or SLAM depth, where gradient is the larger of L1 and L2. These depth are more accurate. 



#### Notes
- Comparison between logL1 and berHu loss.

![](https://cdn-images-1.medium.com/max/1600/1*rmQ8Wmm6ClKAmgty0JmVNg.png)

```python
max_n = 2
xs = np.arange(-max_n, max_n, max_n/100)
delta = 0.2 * max_n
logl1 = np.log(1 + np.abs(xs))
berhu = np.abs(xs) * (np.abs(xs) < delta) + (xs ** 2 + delta ** 2) / (2 * delta) * (np.abs(xs) >= delta)
plt.plot(xs, logl1, label='logl1')
plt.plot(xs, berhu, label='berhu')
plt.legend()
```

- On why BerHu loss is better suited for depth prediction (from FCRN-depth prediction by Laina et al)

>> We provide two further intuitions with respect to
the difference between L2 and berHu loss. In both
datasets that we experimented with, we observe
a heavy-tailed distribution of depth values, also
reported in [27], for which Zwald and LambertLacroix [40] show that the berHu loss function is
more appropriate. This could also explain why
[5, 6] experience better convergence when predicting the log of the depth values, effectively moving a log-normal distribution back to Gaussian.
Secondly we see the greater benefit of berHu in
the small residuals during training as there the L1
derivative is greater than L2’s. This manifests in
the error measures rel. and δ1 (Sec. 4), which are
more sensitive to small errors.

