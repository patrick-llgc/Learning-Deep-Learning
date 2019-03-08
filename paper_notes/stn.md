# [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf)

_Mar 2019_

tl;dr: Create a spatial transformer module to learn invariance to translation, scale, rotation and warping.

#### Overall impression
The STN module transforms data to a canonical, expected pose for easier classification. It can also help localization and is itself a special type of attention.

#### Key ideas
- Three steps in a STN
  - Learn the affine transformation parameters (B x 6) with localization network.
  - Generate sampling grid. `torch.nn.functional.affine_grid`
  - Sample with sampling grid. `torch.nn.functional.grid_sample`
- STN with a more contrained attention transformation (only scale and translation) can be used for weakly supervised localization with only image level labels. This leads to applications such as OCR ([Robust Scene Text Recognition with Automatic Rectification](https://arxiv.org/abs/1603.03915) <kbd>CVPR 2016</kbd>).
- STN is generally only inserted after input. It can also be inerted after the first few convolution layers (when the spatial semantics is still strong) to boost performance.
- Multiple STN can be used together to capture different part of the image for fine grained classification tasks (such as recognizing the different bird species in CUB dataset).

#### Technical details
- The localization network regresses only a few numbers (6 for affine transform and 8 for perspective transform, and NxN for thin-plate-spline transform). This adds to very little computation overhead.
- The input and output of Spatial Transformer has the same number of channels, but may have different spatial resolution.
- The sampling used bilinear interpolation, which can be written out in a surprisingly concise form.

#### Notes
- For co-localization tasks, it seems that this only works if there is only one object in the image.
- Why the transformed image is upright? What enforces it?
  - A: The augmented dataset did not do an isotropic augmentation, but rather a restricted augmentation around the upright postion. Therefore, "the transformation of inputs for all ST models leads to a "standard" upright posed digit -- this is the mean pose found in the training data".
- Bilinear filtering and trilinear filtering.
  - There is [a very nice way](https://en.wikipedia.org/wiki/Bilinear_interpolation#/media/File:Bilinear_interpolation_visualisation.svg) to visualize bilinear interpolation wrt the weights. 
  - [Bilinear filtering](https://en.wikipedia.org/wiki/Bilinear_filtering) works very well when the resampled size is between half and double th original size, beyond which the performance begin to degrade due to alias and missing information. This is because the bilinear interpolation only accounts for the nearest four neighbors.
  - [Trilinear filtering](https://en.wikipedia.org/wiki/Trilinear_filtering) largely resolves aliasing by sampling two nearest image stack from the image pyramid (a [mipmap](https://en.wikipedia.org/wiki/Mipmap)) with bilinear interpolation first and then combine them with a linear interpolation. 
  - Trilinear filtering is not to be confused with trilinear interpolation, which usually applies on a 3D regular grid image
- PyTorch [tutorial](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html). Note that `torch.nn.functional.affine_grid` and `torch.nn.functional.grid_sample` are specifically developed to help build STN faster.
- Tensorflow [implementation](https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py)
- There is [a pytorch repo on thin-plate-spline](https://github.com/WarBean/tps_stn_pytorch) with very interesting visualization.
