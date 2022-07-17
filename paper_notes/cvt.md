# [CVT: Cross-view Transformers for real-time Map-view Semantic Segmentation](https://arxiv.org/abs/2205.02833)

_July 2022_

tl;dr: Dense prediction version of PETR.

#### Overall impression
The paper proposed the idea of 3D positional embedding to help cross attention module learn the correspondence between image view and BEV view. This idea is very much like that of [PETR](petr.md) series. [CVT](cvt.md) does not use sparse object query to learn end to end object detection. Instead, CVT uses cross attention and 3D PE to transform perspective features into BEV features, and attach conventional object detection to BEV features. 

Each camera uses optional embeddings that depend on its intrinsics and extrinsic calibration.

The performance is roughly on par with [FIERY](fiery.md) static, but is much simpler, easier to train and easier to deploy. This proves that CVT combines features in a more efficient manner. 

#### Key ideas
- The cross-view attention is to link up a map-view representation with image-view features. CVT does not learn an explicit estimate of depth but encode any depth ambiguity in the positional embeddings and let a transformer lean a proxy for depth.
- cross view attention
	- K: positional embedding + image features
	- V: image features
	- Q: map view queries, learned BEV PE + camera extrinsics
- Multi-resolution patch embedding: 2 is enough.

#### Technical details
- **Why do we need PE?** Image feature alone is hard to link IV and BEV, as the network needs to explicitly infer the direction each image is facing to disambiguate diff views.
- All cameras in CVT **share the same image-encoder**, but uses a positional embedding dependent on the individual camera calibration.
- MLP methods (such as [VPN](vpn.md)) forego the inherit inductive biases contained in a calibrated camera setup and instead need to learn an implicit model of camera calibration baked into the network weights. --> Ablation study shows that **a learned embedding per camera works surprisingly well** (without only 1.5% drop in mIOU) as most of the time the calibration is static. However **this may face issues in production** when calibration from different cars are different, as the MLP weight can only bake the calibration of a single set up into the weights. 
- [OFT](oft.md) foregoes an explicit depth estimate and instead averages all possible image locations a map-view object could take. (Average smear, like backprojection in X-ray recon.)
- PON settings (50x100 m, 0.25 m/pixel), LSS setting (100x100 m, 0.5 m/pixel).
- Transformer based methods are typically robust to camera dropout, and overall performance does not degrade beyond unobserved part of the scenes. For example, if we drop the rear cam, the performance in the front should be largely the same.

#### Notes
- [Github](https://github.com/bradyz/cross_view_transformers)
- We should do the same for visualization of the cross-view attention.