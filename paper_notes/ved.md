# [VED: Monocular Semantic Occupancy Grid Mapping with Convolutional Variational Encoder-Decoder Networks](https://arxiv.org/abs/1804.02176)

_September 2020_

tl;dr: Use variational autoencoder for semantic occpuancy grid map prediction.

#### Overall impression
Variational encoder-decoder (VED) encodes the front-view visual information for he driving scene and subsequently decodes it into a BEV semantic occupancy grid. 

The proposed method beats a vanilla SegNet (a relatively strong baseline for conventional semantic segmentation). There was a 2x1 pooling layer in order to accommodate the different aspect ratio of input and output. 

#### Key ideas
- Binary occupancy grid is a decades old concept, but semantic occupancy grid is more powerful and enables more efficient and reliable navigation.
- Variational AutoEncoder (VAE, or VED as referred to in this paper) forces the latent space to be a normal distribution. Thus we can add a KL-divergence loss to encourage the latent distribution to be a normal distribution. The paper mainly wants to exploits VED's sampling robustness to imperfect GT.
- VED exhibits intrinsic invariance wrt pitch and roll perturbations, compared to monocular baseline and flat ground assumption.

#### Technical details
- It is more robust to pitch and roll perturbation. It can also generalize better to unseen scenario.
- The PCA components of the latent space does encode some interpretable results. 

#### Notes
- Questions and notes on how to improve/revise the current work  

