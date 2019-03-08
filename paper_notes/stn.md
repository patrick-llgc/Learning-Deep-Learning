# [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf)

_Mar 2019_

tl;dr: Create a spatial transformer module to learn invariance to translation, scale, rotation and warping.

#### Overall impression
The STN module transforms data to a canonical, expected pose for easier classification. It can also help localization and is itself a special type of attention.

#### Key ideas
- Three steps in a STN
- Learn the affine transformation parameters (B x 6).
- Generate sampling grid. `torch.nn.functional.affine_grid`
- Sample with sampling grid. `torch.nn.functional.grid_sample`

#### Technical details
- Summary of technical details

#### Notes
- PyTorch [tutorial](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html). Note that `torch.nn.functional.affine_grid` and `torch.nn.functional.grid_sample` are specifically developed to help build STN faster.
- Tensorflow [implementation](https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py)